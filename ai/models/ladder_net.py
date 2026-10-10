"""LadderNet — one backbone that can express every rung of P21.

`research/plans/P21_architecture_ladder.md` builds the architecture up from a flat MLP one
mechanism at a time, so that each mechanism gets a matched control by construction. That needs a
backbone whose structure is *configuration*, not a class hierarchy:

    M0  input_mode="flat"                                     a plain MLP
    M1  input_mode="grouped"                                  per-block dense projections
    M2  input_mode="entity",  aggregation="flatten"           shared encoder, position kept
    M3  ... + card_self_attention=True                        cards attend to cards
    M4  ... + cross_attention=True                            cards attend to countries
    M5  ... aggregation="pool"                                the removal that is measured last

and orthogonally `per_entity_heads` and `identity_dim`, which exist to repair the pooling loss
and are therefore tested as a 2x2 against `aggregation` rather than as rungs.

**Every axis is required at construction. There are no defaults.** A defaulted `layout` argument
was the mechanism behind five separate instances of one bug in this repository, because handing a
model the wrong variant returns a number instead of raising. The same discipline applies here: the
configuration is explicit, it is recorded on the model by `ladder_config()`, and `create_like`
rebuilds a copy from that record rather than from a caller's argument list.

`ColdWarNetMLP` already implements the M0 shape and predates this file; `input_mode="flat"`
reproduces it. The older class is left alone so E3-era MLP checkpoints keep loading.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.coldwar_net_v2 import (STATIC_BOARD_SLOTS, STATIC_CARD_SLOTS,
                                      ColdWarNetV2, static_input_mask)
from bindings.action_encoder import ActionEncoder

#: The branch block of the flat action space (event branches, CONFIRM_DONE, DEFCON values, regions).
BRANCH_BLOCK = ActionEncoder.FLAT_ACTION_SIZE - ActionEncoder.BRANCH_OFFSET
#: The play-mode block (EVENT, SPACE, OPS_INFLUENCE, OPS_COUP, OPS_REALIGN).
PLAY_MODE_BLOCK = 5
#: Card-feature slot 13: 1.0 for the card the decision is about (engine/include/ts/game_state.hpp).
ACTIVE_CARD_SLOT = 13

#: Countries the ownership head predicts: all 84 (P29 bet 2).
#: Width of each optional observation block, by ts.OBS_FEATURE_* bit (engine/include/ts/game_state.hpp,
#: obs_features). Restated so the model does not import the engine; tests/bindings check it
#: against ts.obs_size_for.
OBS_FEATURE_WIDTHS: Dict[int, int] = {1 << 1: 3}   # OPS_BUDGET


def _obs_extra(features: int) -> int:
    unknown = int(features) & ~sum(OBS_FEATURE_WIDTHS)
    if unknown:
        raise ValueError(f"unknown observation feature bits {unknown:#x}")
    return sum(w for bit, w in OBS_FEATURE_WIDTHS.items() if int(features) & bit)


AUX_OWN_COUNTRIES = 84
#: outputs per card of the card-event head: 5 Ops-reach + 12 event-outcome targets
#: (ai.training.card_event_targets.AUX_DIM; restated here so the model does not import the engine).
CARD_AUX_DIM = 17
#: Attention heads in the C1 token path. Pinned, like the lookup's head width, because the head
#: count cannot be recovered from the weights.
TOKEN_HEADS = 4
#: How the observation is read before the trunk.
INPUT_MODES: Tuple[str, ...] = ("flat", "grouped", "entity")
#: How per-entity tokens become a fixed-size vector. Only meaningful for `input_mode="entity"`.
AGGREGATIONS: Tuple[str, ...] = ("flatten", "pool")
#: Which per-entity heads exist. The card-collision finding predicts `country` carries most of
#: M2's +352.8 Elo, since `pe_card` cannot distinguish 95 of 110 cards without identity.
HEAD_ENTITIES: Tuple[str, ...] = ("both", "country", "card")


class _LeakUp(torch.autograd.Function):
    """Adds zero going forward; going back, `leak` times the incoming gradient where it is negative
    -- where gradient descent would RAISE the logit -- and nothing where it would lower it."""

    @staticmethod
    def forward(ctx: Any, z: torch.Tensor, leak: float) -> torch.Tensor:   # type: ignore[override]
        ctx.leak = float(leak)
        return torch.zeros_like(z)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[torch.Tensor, None]:   # type: ignore[override]
        return ctx.leak * grad.clamp(max=0.0), None


class LadderNet(ColdWarNetV2):
    """A P21 rung. All structural axes are explicit; see the module docstring."""

    # The M2 family may switch either head off (`head_entities`) or remove the trunk context
    # (`head_context`), so these are optional here where the base class always builds them.
    pe_trunk: nn.Linear | None          # type: ignore[assignment]
    pe_country: nn.Sequential | None    # type: ignore[assignment]
    pe_card: nn.Sequential | None       # type: ignore[assignment]

    def __init__(self, *,
                 input_mode: str,
                 aggregation: str,
                 entity_dim: int,
                 card_self_attention: bool,
                 cross_attention: bool,
                 per_entity_heads: int,
                 head_context: bool,
                 head_static: bool,
                 head_entities: str,
                 identity_dim: int,
                 card_lookup: bool,
                 card_lookup_heads: int,
                 card_lookup_dim: int,
                 card_lookup_identity_dim: int,
                 drop_static: bool,
                 hidden_dim: int,
                 num_res_blocks: int,
                 entity_proj_dim: int,
                 num_attn_heads: int,
                 categorical_value: bool,
                 head_center: bool = False,
                 aux_heads: bool = False,
                 card_aux: bool = False,
                 opp_legal_aux: bool = False,
                 branch_head: bool = False,
                 play_mode_head: bool = False,
                 obs_features: int = 0,
                 logit_cap: float = 0.0,
                 logit_cap_grad: str = "tanh",
                 logit_cap_leak: float = 0.0,
                 logit_cap_leak_up: float = 0.0,
                 token_layers: int = 0,
                 token_dim: int = 0,
                 **kwargs: Any) -> None:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}; got {input_mode!r}")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}; got {aggregation!r}")
        if head_entities not in HEAD_ENTITIES:
            raise ValueError(f"head_entities must be one of {HEAD_ENTITIES}; "
                             f"got {head_entities!r}")
        if not per_entity_heads and (not head_context or not head_static
                                     or head_entities != "both"):
            raise ValueError(
                "head_context / head_static / head_entities only mean anything with "
                "--per-entity-heads > 0. Refused rather than silently ignored.")

        if token_layers and input_mode != "grouped":
            raise ValueError("token_layers is a parallel path beside the grouped projections; "
                             "it needs input_mode='grouped'.")
        if token_layers and token_dim <= 0:
            raise ValueError(f"token_layers={token_layers} needs a positive token_dim")
        tokenless = input_mode in ("flat", "grouped")
        # `grouped` keeps every entity's RAW slots at a fixed offset, so a per-entity head can
        # read country i's own features and reach country i's logit with no learned token space
        # in between -- `pe_country` already takes the raw slots alongside the token. That is the
        # lookup mechanism on its own, uncontaminated by the lossy 26->d compression that a
        # shared encoder imposes.
        raw_tokens = (input_mode == "grouped") and bool(per_entity_heads)
        if tokenless:
            # Refused rather than ignored. A model built with heads it cannot feed would train
            # happily and silently be a different architecture from the one that was asked for.
            if per_entity_heads and input_mode == "flat":
                raise ValueError(
                    "input_mode='flat' never reshapes the observation into entities, so "
                    "per-entity heads have nothing to read. Use 'grouped' for raw-feature "
                    "per-entity heads.")
            if card_self_attention or cross_attention:
                raise ValueError(
                    f"input_mode={input_mode!r} forms no attention tokens. Cross-attention "
                    f"between 14-wide card rows and 26-wide country rows needs kdim/vdim "
                    f"plumbing and a head count dividing 14; that is a separate rung.")
            if identity_dim and not raw_tokens:
                raise ValueError(
                    f"input_mode={input_mode!r} reads each entity at a fixed offset, so position "
                    f"already identifies it for the trunk. An identity embedding is only useful "
                    f"where a function is SHARED across entities -- add --per-entity-heads and "
                    f"it will be given to that head.")
        else:
            # P21: static per-entity slots are what a *shared* encoder uses to tell its tokens
            # apart. Dropping them is only correct where the reader is positional.
            if drop_static:
                raise ValueError(
                    "drop_static is for positional readers (flat/grouped), where a constant "
                    "input is exactly a bias. With a shared per-entity encoder the static slots "
                    "are how it knows what kind of entity it is looking at.")

        super().__init__(hidden_dim=hidden_dim, num_res_blocks=num_res_blocks,
                         num_attn_heads=num_attn_heads, categorical_value=categorical_value,
                         # Built for a raw-token head too: there the identity vector is the
                         # head's only way to tell two same-typed entities apart.
                         identity_dim=(identity_dim if (not tokenless or raw_tokens) else 0),
                         per_entity_heads=0,          # rebuilt below at the right token width
                         graph_layers=0,              # P21 runs without graph convolution
                         self_transform=False,
                         attn_readout=0,
                         # The view spec (owner, 2026-10-01): optional observation blocks are
                         # appended right after the 100 globals, so they widen the global slice
                         # that `global_proj` reads and nothing else moves.
                         global_features=ColdWarNetV2.GLOBAL_SIZE + _obs_extra(obs_features),
                         **kwargs)

        self.input_mode = str(input_mode)
        self.aggregation = str(aggregation)
        # Canonical: the token width is meaningless without tokens, so a tokenless
        # rung records 0. That makes `ladder_config()` exactly recoverable from the
        # weights, which is what `ladder_config_from_state_dict` relies on.
        self.entity_dim = 0 if tokenless else int(entity_dim)
        self.card_self_attention = bool(card_self_attention)
        self.cross_attention = bool(cross_attention)
        self.drop_static = bool(drop_static)
        self.raw_tokens = bool(raw_tokens)
        self.head_context = bool(head_context)
        self.head_static = bool(head_static)
        self.head_entities = str(head_entities)
        self.entity_proj_dim = int(entity_proj_dim)
        self.ladder_num_attn_heads = int(num_attn_heads)
        self.ladder_per_entity_heads = int(per_entity_heads)
        # Kept explicitly rather than read back off `fusion_in`, so the config
        # record cannot drift from what was asked for.
        self.ladder_hidden_dim = int(hidden_dim)

        # Everything v2 builds that this configuration does not use is *replaced*, not bypassed:
        # a registered-but-unused module puts never-updated parameters into the checkpoint and
        # into any count of how big the network is, which is the number the ladder exists to keep
        # honest.
        for name in ("gconv1", "gconv2", "board_fc", "board_proj", "card_fc", "card_proj",
                     "cross_attn", "cross_attn_ln", "cross_card_proj", "hist_conv",
                     "ro_query", "ro_country_kv", "ro_card_kv", "ro_out"):
            if hasattr(self, name):
                setattr(self, name, None)

        d = int(entity_dim)
        p = self.entity_proj_dim
        # The trunk reads raw slots in tokenless modes; identity, when present, goes to the
        # head alone. In `entity` mode it is concatenated before the shared encoder as usual.
        board_in = self.board_features + (0 if tokenless else self.identity_dim)
        card_in = self.card_features + (0 if tokenless else self.identity_dim)

        if self.input_mode == "flat":
            keep = ~static_input_mask(self.board_features, self.card_features,
                                      self.TOTAL_OBS_SIZE) if self.drop_static \
                else torch.ones(self.TOTAL_OBS_SIZE, dtype=torch.bool)
            self.register_buffer("keep_idx", torch.nonzero(keep).squeeze(-1), persistent=False)
            self.lad_in = _mlp(int(keep.sum()), p)
            fused_width = p

        elif self.input_mode == "grouped":
            if self.drop_static:
                keep = ~static_input_mask(self.board_features, self.card_features,
                                          self.TOTAL_OBS_SIZE)
            else:
                keep = torch.ones(self.TOTAL_OBS_SIZE, dtype=torch.bool)
            b_keep = torch.nonzero(keep[:self.BOARD_SIZE]).squeeze(-1)
            c_keep = torch.nonzero(
                keep[self.CARD_OFFSET:self.CARD_OFFSET + self.CARD_SIZE]).squeeze(-1)
            self.register_buffer("board_keep_idx", b_keep, persistent=False)
            self.register_buffer("card_keep_idx", c_keep, persistent=False)
            self.lad_board = _mlp(int(b_keep.numel()), p)
            self.lad_card = _mlp(int(c_keep.numel()), p)
            fused_width = 2 * p + 128

        else:  # entity
            self.lad_board_enc = nn.Sequential(nn.Linear(board_in, d), nn.GELU())
            self.lad_card_enc = nn.Sequential(nn.Linear(card_in, d), nn.GELU())
            if self.card_self_attention:
                self.lad_self_attn = nn.MultiheadAttention(
                    embed_dim=d, num_heads=num_attn_heads, batch_first=True)
                self.lad_self_ln = nn.LayerNorm(d)
            if self.cross_attention:
                self.lad_cross_attn = nn.MultiheadAttention(
                    embed_dim=d, num_heads=num_attn_heads, batch_first=True)
                self.lad_cross_ln = nn.LayerNorm(d)
            board_agg = 84 * d if self.aggregation == "flatten" else 2 * d
            card_agg = 110 * d if self.aggregation == "flatten" else 2 * d
            self.lad_board = _mlp(board_agg, p)
            self.lad_card = _mlp(card_agg, p)
            if self.cross_attention:
                self.lad_cross = _mlp(card_agg, p)
            fused_width = (3 if self.cross_attention else 2) * p + 128

        # P22: identity-keyed card lookup. A parallel path over the RAW card rows -- the trunk's
        # own card projection is untouched, so this adds a mechanism rather than replacing one.
        self.card_lookup = bool(card_lookup)
        self.card_lookup_heads = int(card_lookup_heads)
        self.card_lookup_dim = int(card_lookup_dim)
        self.card_lookup_identity_dim = int(card_lookup_identity_dim)
        self.cl_identity: nn.Parameter | None = None
        self.cl_key: nn.Linear | None = None
        self.cl_value: nn.Linear | None = None
        self.cl_query: nn.Linear | None = None
        self.cl_out: nn.Linear | None = None
        if self.card_lookup:
            if self.card_lookup_heads <= 0 or self.card_lookup_dim <= 0:
                raise ValueError(
                    f"card_lookup needs positive heads and dim, got "
                    f"{self.card_lookup_heads} and {self.card_lookup_dim}.")
            hk = self.card_lookup_heads * self.card_lookup_dim
            if self.card_lookup_identity_dim:
                # Identity is what makes a card addressable. Without it, Europe Scoring and Asia
                # Scoring are identical on every property -- same ops, same era, both scoring --
                # so a query retrieves an average over the cards it needed to tell apart. The
                # identity-free variant is kept reachable because it is the ablation that
                # attributes the gain, not because it is expected to work.
                # Scaled to UNIT expected norm, not the usual small embedding init. The keys
                # concatenate identity with the raw row, whose slots are 0/1 with a norm around
                # 1.5, so an identity at 0.02 is ~50x smaller and simply invisible beside them:
                # at that scale a query built from card 12's own identity ranked it 107th of 110.
                # At 1/sqrt(d) it ranks 1st. Training could in principle grow the embedding, but
                # starting a mechanism in a regime where it cannot be used is not a fair test of
                # it.
                self.cl_identity = nn.Parameter(
                    torch.randn(110, self.card_lookup_identity_dim)
                    / (self.card_lookup_identity_dim ** 0.5))
            # Keys carry the FULL row, location included, not just the properties. With location
            # only in the values the query can ask "where is card X" but cannot select BY
            # location, so "do I hold cards like Y" is inexpressible -- the query has nothing to
            # match a location against. Both query types need it, so it appears in keys and
            # values alike.
            self.cl_key = nn.Linear(self.card_lookup_identity_dim + self.card_features, hk)
            self.cl_value = nn.Linear(self.card_features, hk)
            # Queried from the PRE-fusion vector, not the trunk: the output is concatenated into
            # fusion_in's input, so querying the trunk would be circular.
            self.cl_query = nn.Linear(fused_width, hk)
            self.cl_out = nn.Linear(hk, hk)
            fused_width += hk

        # P30 C1: card and country tokens with attention, beside the grouped projections. Every
        # country row and card row becomes a token (a projection of its full row plus a learned
        # identity), the globals one more; `token_layers` pre-norm transformer layers let any
        # token read any other -- the card x board interactions the grouped trunk represents only
        # partly (research/log/P30_card_board_targets.md: 0.80 against 0.90 with attention). The
        # global token's output joins the fusion input, and each per-entity head reads its own
        # contextualised token. Canonical 0/0 when off, so the config is recovered from weights.
        self.token_layers = int(token_layers)
        self.token_dim = int(token_dim) if self.token_layers else 0
        if self.token_layers:
            t = self.token_dim
            self.tok_country_in = nn.Linear(self.board_features, t)
            self.tok_card_in = nn.Linear(self.card_features, t)
            self.tok_global_in = nn.Linear(self.GLOBAL_SIZE, t)
            self.tok_country_id = nn.Parameter(torch.randn(84, t) * 0.02)
            self.tok_card_id = nn.Parameter(torch.randn(110, t) * 0.02)
            layer = nn.TransformerEncoderLayer(t, TOKEN_HEADS, 2 * t, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.tok_enc = nn.TransformerEncoder(layer, self.token_layers,
                                                 enable_nested_tensor=False)
            self.tok_norm = nn.LayerNorm(t)
            fused_width += t

        self.fusion_in = nn.Sequential(
            nn.Linear(fused_width, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        # Per-entity heads at *this* token width. v2 hard-codes 64, which would silently build
        # the wrong shape for any other entity_dim.
        self.per_entity_heads = int(per_entity_heads)
        if self.per_entity_heads:
            k = self.per_entity_heads
            # `ctx` is the ONLY path from the trunk into the correction. Without it the head is a
            # pure per-entity map and the whole mechanism is embarrassingly parallel (arm M2a).
            ctx_w = k if self.head_context else 0
            self.pe_trunk = nn.Linear(hidden_dim, k) if self.head_context else None
            # With raw tokens the raw slots arrive in the `raw` slot, so the token slot carries
            # the identity vector instead -- width 0 when there is no identity.
            tok_w = (self.identity_dim + self.token_dim) if self.raw_tokens else d
            # Dropping the constant slots leaves the dynamic ones, which is where the mechanism's
            # content has to be: constants alone would be a fixed per-TYPE bias (arm M2b).
            b_raw = board_in - (0 if self.head_static else len(STATIC_BOARD_SLOTS))
            c_raw = card_in - (0 if self.head_static else len(STATIC_CARD_SLOTS))
            self.pe_country = (nn.Sequential(nn.Linear(tok_w + b_raw + ctx_w, k), nn.GELU(),
                                             nn.Linear(k, 1))
                               if self.head_entities in ("both", "country") else None)
            self.pe_card = (nn.Sequential(nn.Linear(tok_w + c_raw + ctx_w, k), nn.GELU(),
                                          nn.Linear(k, 1))
                            if self.head_entities in ("both", "card") else None)
            for head in (self.pe_country, self.pe_card):
                if head is None:
                    continue
                out = head[-1]
                assert isinstance(out, nn.Linear)
                nn.init.zeros_(out.weight)
                nn.init.zeros_(out.bias)
        # --ladder-head-center: the per-entity heads' hidden features are centred across entities
        # before the final projection, so a head's output has no common component except its
        # final bias. In E4 a decision's legal set never mixes countries with non-country actions,
        # so a shift common to every country logit is invisible to the policy and gets no gradient;
        # left free it drifts without limit -- to a median ~2,300 and a tail of ~40,000 by 590M in
        # E4-57-44, all of it in pe_country's output (research/log/E4_long_runs.md) -- and at that
        # scale TF32 rounding costs nats on unlikely actions. Centring the features rather than the
        # output keeps the raw values at the size of the differences between entities, and the
        # bias, the only common term left, receives exactly zero gradient and never moves. The
        # function class is unchanged up to that invisible shift. E4.1's merged view does mix
        # countries with play modes, so the trainer refuses the combination. Recorded as a buffer,
        # so the architecture is recovered from the weights like everything else.
        if head_center and not self.per_entity_heads:
            raise ValueError("head_center centres the per-entity heads; there are none to centre.")
        self.head_center = bool(head_center)
        if self.head_center:
            self.register_buffer("pe_center", torch.ones(()))

        # P29 bet 2: auxiliary targets read off the trunk, trained from each game's end -- who
        # controls every country (mine / the opponent's / neither, in the mover's frame) and the
        # final VP margin. KataGo's ownership and score heads: the trunk learns what holding a
        # country is worth before the policy can take it. Used by the training loss only, never
        # by the policy or the value; forward() is untouched. Built only when asked for, and
        # recovered from the weights (`aux_own_head.*`) like every other optional part.
        self.aux_heads = bool(aux_heads)
        if self.aux_heads:
            self.aux_own_head = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.GELU(), nn.Linear(256, AUX_OWN_COUNTRIES * 3))
            self.aux_vp_head = nn.Sequential(
                nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 1))

        # P30: the card-event auxiliary target (--aux-card-events). For every card, what its event
        # would do on this board and what its Ops could take (ai.training.card_event_targets),
        # read off the trunk so the gradient reaches it: the trained trunks were found to carry no
        # more of this than an untrained one (research/log/P30_card_board_targets.md). Training
        # loss only; forward() is untouched. Recovered from the weights (`card_aux_head.*`).
        # The observation feature set (ts.OBS_FEATURE_* bits) this network reads, kept in the
        # weights so a checkpoint names its own view: width alone cannot tell two equal-width
        # feature sets apart. A buffer only when non-zero, so base checkpoints are unchanged.
        self.obs_feature_bits = int(obs_features)
        if self.obs_feature_bits:
            self.register_buffer("obs_features", torch.tensor(self.obs_feature_bits, dtype=torch.int64))
        # --ladder-logit-cap (owner, 2026-10-09): each legal move's deficit to the top legal move is
        # bounded to c through c * tanh(deficit / c). Uncapped, the policy saturates -- a coup under
        # the opponent's Cuban Missile Crisis led influence by 28 nats, P(coup) was 1 in float32, and
        # policy gradient, scaled by 1 - P, could not move it however the game ended
        # (research/log/E7_cuban_missile_crisis_probe.md). Centred on the top move, not the mean:
        # the mean is dragged down by the many hopeless legal moves, and centring there squashes
        # the good moves together. Monotonic, so the greedy move is unchanged. A buffer only when
        # on, so uncapped checkpoints are unchanged.
        self.logit_cap_value = float(logit_cap)
        # How the gradient passes the cap. "tanh": through it -- d/dz of c*tanh(d/c) is
        # 1 - tanh^2(d/c), ~0.0013 at a 28-nat deficit, so a saturated move's raw logit learns
        # ~750x slower than its capped probability says, and moving the top logit moves every
        # saturated capped logit with it. "straight": the capped values forward, the raw logits'
        # gradient as if uncapped (straight-through) -- a sampled alternative is pushed at the
        # full softmax gradient. Forward, and so play, is identical; only training differs.
        if logit_cap_grad not in ("tanh", "straight"):
            raise ValueError(f"logit_cap_grad must be 'tanh' or 'straight', not {logit_cap_grad!r}")
        self.logit_cap_grad = str(logit_cap_grad)
        # A middle way: the tanh's own gradient plus `leak` times the raw one (out = capped +
        # leak * (z - z.detach())), so a saturated move gets ~leak of the full softmax gradient
        # while the tanh still damps sharpening. The straight-through cap removed that damping and
        # the policy sharpened fast (entropy 0.31 -> 0.22, KL per update 0.1-0.2 within 15M,
        # E7-A8-R1-S44@6400M+A10). Forward, and so play, is unchanged.
        self.logit_cap_leak = float(logit_cap_leak)
        # The leak only where it raises a logit (the incoming gradient negative). A capped policy
        # cannot push a bad move below the floor (~e^-c), so a gradient pushing saturated moves
        # DOWN never stops; leaked into the trunk it drifts the whole network -- the symmetric
        # leak did (A11: entropy 0.37 -> 0.28, KL to pi_ref 0.008 -> 0.046 in 40M). Pushes UP are
        # satisfiable -- a raised move enters the cap's band and the ordinary softmax takes over --
        # and they are the ones a trap needs: a losing coup's negative advantage raises the rest.
        self.logit_cap_leak_up = float(logit_cap_leak_up)
        if self.logit_cap_value > 0.0:
            self.register_buffer("logit_cap", torch.tensor(self.logit_cap_value, dtype=torch.float32))
        self.card_aux = bool(card_aux)
        if self.card_aux:
            self.card_aux_head = nn.Sequential(
                nn.Linear(hidden_dim, 512), nn.GELU(), nn.Linear(512, 110 * CARD_AUX_DIM))
        # P31 1d (--aux-opp-legality): per country, whether the OPPONENT may place influence there
        # and may coup there at its next decision -- after this side's play, so it is not the
        # current observation's can_opp_place / can_opp_coup (slots 20, 22), which a head would
        # merely copy. It teaches the trunk what a play does to the opponent's options (Chernobyl's
        # region, DEFCON's locks). Training loss only; forward() is untouched. Recovered from the
        # weights (`opp_legal_head.*`).
        self.opp_legal_aux = bool(opp_legal_aux)
        if self.opp_legal_aux:
            self.opp_legal_head = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.GELU(), nn.Linear(256, 84 * 2))
        # --ladder-branch-head (owner, 2026-10-06): the branch block's logits (event branches,
        # CONFIRM_DONE, DEFCON values, regions -- flat 200..219) get a correction read from the trunk
        # AND which card is resolving (the observation's ACTIVE_NOW card flag). The branch slots are
        # shared by every card with a branch, so slot 0 is "end the game" for Wargames and something
        # else elsewhere; a head without the card learns a card-independent average. Measured: the
        # trunk predicts whether ending at Wargames' branch wins at 0.98 AUC while the plain policy's
        # P(end) is ~0.2 at any lead (research/log/P31_branch_arms_B1_B2.md). The last layer starts at
        # zero, so a network with the head starts as the one without it.
        self.branch_head = bool(branch_head)
        if self.branch_head:
            if input_mode == "flat":
                raise ValueError("branch_head reads the card nodes; the flat input mode has none")
            self.branch_head_net = nn.Sequential(
                nn.Linear(hidden_dim + 110, 128), nn.GELU(), nn.Linear(128, BRANCH_BLOCK))
            out = self.branch_head_net[-1]
            assert isinstance(out, nn.Linear)
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        # --ladder-play-mode-head (owner, 2026-10-07): the same for the play-mode block (flat 110..114:
        # EVENT, SPACE, OPS_INFLUENCE / COUP / REALIGN -- the last three also the deferred Ops-mode
        # choice), read from the trunk and the card being played (ACTIVE_NOW at those decisions). The
        # play-mode slots are shared by all 110 cards, so a push on one card's event moves every card's
        # (E7-29-44: forcing three cards' events halved the event share of every play-mode decision).
        # Zero-initialised.
        self.play_mode_head = bool(play_mode_head)
        if self.play_mode_head:
            if input_mode == "flat":
                raise ValueError("play_mode_head reads the card nodes; the flat input mode has none")
            self.play_mode_head_net = nn.Sequential(
                nn.Linear(hidden_dim + 110, 128), nn.GELU(), nn.Linear(128, PLAY_MODE_BLOCK))
            out = self.play_mode_head_net[-1]
            assert isinstance(out, nn.Linear)
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)

    def forward_card_aux(self, obs: torch.Tensor) -> torch.Tensor:
        """The card-event predictions, (B, 110, CARD_AUX_DIM): per card, the standardised
        ops-arithmetic and event-outcome targets of `ai.training.card_event_targets`."""
        if not self.card_aux:
            raise RuntimeError("this network was built without the card-event head (--aux-card-events)")
        h, _attn, _tokens = self._encode(obs)
        return self.card_aux_head(h).view(-1, 110, CARD_AUX_DIM)

    def forward_opp_legal(self, obs: torch.Tensor) -> torch.Tensor:
        """The opponent-legality logits, (B, 84, 2): per country, may the opponent place influence
        there (0) and coup there (1) at its next decision."""
        if not self.opp_legal_aux:
            raise RuntimeError("this network was built without the opponent-legality head (--aux-opp-legality)")
        h, _attn, _tokens = self._encode(obs)
        return self.opp_legal_head(h).view(-1, 84, 2)

    def forward_aux(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The auxiliary predictions: ownership logits (B, 84, 3) -- mine / opponent's / neither --
        and the final VP margin (B, 1) in the mover's frame, VP / 20."""
        if not self.aux_heads:
            raise RuntimeError("this network was built without aux heads (--aux-ownership / --aux-vp-margin)")
        h, _attn, _tokens = self._encode(obs)
        own = self.aux_own_head(h).view(-1, AUX_OWN_COUNTRIES, 3)
        return own, self.aux_vp_head(h)


    def _card_lookup(self, card_raw: torch.Tensor, pre: torch.Tensor,
                     b: int) -> torch.Tensor:
        """Content-addressed retrieval over the 110 card rows.

        Keys and values both carry the full row, so two question shapes are expressible:

          * **where is card X** -- the query matches an identity, and the retrieved value's
            location slots answer it;
          * **do I hold cards like Y** -- the query matches a location together with properties,
            which needs location in the KEYS; with it only in the values there is nothing for a
            query to select on.

        The query comes from `pre`, so either question is conditional on state -- *Europe is
        negative, therefore check Europe Scoring* -- where the dense card projection computes
        every lookup unconditionally.
        """
        assert self.cl_key is not None and self.cl_value is not None
        assert self.cl_query is not None and self.cl_out is not None
        rows = card_raw.view(b, 110, self.card_features)
        if self.cl_identity is not None:
            ident = self.cl_identity.unsqueeze(0).expand(b, -1, -1)
            k_in = torch.cat([ident, rows], dim=-1)
        else:
            k_in = rows
        nh, dk = self.card_lookup_heads, self.card_lookup_dim
        k = self.cl_key(k_in).view(b, 110, nh, dk).transpose(1, 2)    # (B,nh,110,dk)
        v = self.cl_value(rows).view(b, 110, nh, dk).transpose(1, 2)  # (B,nh,110,dk)
        q = self.cl_query(pre).view(b, nh, 1, dk)                     # (B,nh,1,dk)
        w = torch.softmax((q @ k.transpose(-2, -1)) / (dk ** 0.5), dim=-1)
        return self.cl_out((w @ v).view(b, nh * dk))

    def _token_path(self, board_raw: torch.Tensor, card_raw: torch.Tensor, glob: torch.Tensor,
                    b: int) -> torch.Tensor:
        """(B, 1 + 84 + 110, token_dim): the global token, then countries, then cards, after
        `token_layers` of full self-attention."""
        # bf16 on the GPU: the 195-token attention is the dominant cost of the whole network
        # (~3.5x faster than TF32 measured at batch 4096), and its output re-enters fp32 at the
        # fusion. Off the GPU (CPU tournaments, ONNX export) it runs in fp32.
        with torch.autocast(device_type=board_raw.device.type, dtype=torch.bfloat16,
                            enabled=board_raw.is_cuda):
            c = self.tok_country_in(board_raw.view(b, 84, self.board_features)) + self.tok_country_id
            k = self.tok_card_in(card_raw.view(b, 110, self.card_features)) + self.tok_card_id
            g = self.tok_global_in(glob).unsqueeze(1)
            out = self.tok_norm(self.tok_enc(torch.cat([g, c, k], dim=1)))
        return out.float()

    # ------------------------------------------------------------------ config

    def _entity_out(self, head: nn.Module, x: torch.Tensor,
                    ctx: torch.Tensor | None = None) -> torch.Tensor:
        """A per-entity head over [batch, entities, features] -> [batch, entities]; under
        head_center the hidden features are centred across entities before the last layer.

        `ctx` [batch, k], when given, is the trunk context that every entity of a sample shares,
        read by the last k input columns of the head's first layer. Its term is computed once per
        sample and broadcast over the entities instead of being concatenated onto each of them:
        Linear(cat[x, ctx]) = x Wx^T + (ctx Wc^T + b) is the same function from the same weights,
        without 84 (or 110) copies of one product. Concatenated, the 90-wide input also missed
        the tensor cores (not 8-aligned) and the head cost more than the rest of the network; split,
        E7's update fell from 0.36 to 0.29 s per iteration, and the two forms agree to 3e-13 in
        float64 (research/log/E7_training_throughput_profile.md).
        """
        assert isinstance(head, nn.Sequential)
        if ctx is None:
            f = head[:-1](x)
        else:
            first = head[0]
            assert isinstance(first, nn.Linear)
            kx = x.shape[-1]
            pre = (F.linear(x, first.weight[:, :kx])
                   + F.linear(ctx, first.weight[:, kx:], first.bias).unsqueeze(1))
            f = head[1:-1](pre)
        if self.head_center:
            f = f - f.mean(dim=1, keepdim=True)
        return head[-1](f).squeeze(-1)

    def ladder_config(self) -> Dict[str, Any]:
        """The full structural configuration, for the checkpoint and for `create_like`.

        Read off the model, never from a caller's arguments: a frozen evaluation copy built by
        listing arguments has to list all of them, and this repository has twice lost a run at
        its first snapshot because one was left at a factory default.
        """
        return dict(
            input_mode=self.input_mode,
            aggregation=self.aggregation,
            entity_dim=self.entity_dim,
            card_self_attention=self.card_self_attention,
            cross_attention=self.cross_attention,
            per_entity_heads=self.per_entity_heads,
            head_context=self.head_context,
            head_static=self.head_static,
            head_entities=self.head_entities,
            identity_dim=self.identity_dim,
            card_lookup=self.card_lookup,
            card_lookup_heads=self.card_lookup_heads,
            card_lookup_dim=self.card_lookup_dim,
            card_lookup_identity_dim=self.card_lookup_identity_dim,
            token_layers=self.token_layers,
            token_dim=self.token_dim,
            drop_static=self.drop_static,
            hidden_dim=self.ladder_hidden_dim,
            num_res_blocks=len(self.res_blocks),
            entity_proj_dim=self.entity_proj_dim,
            num_attn_heads=self.ladder_num_attn_heads,
            categorical_value=bool(self.categorical_value),
            head_center=self.head_center,
            aux_heads=self.aux_heads,
            card_aux=self.card_aux,
            opp_legal_aux=self.opp_legal_aux,
            branch_head=self.branch_head,
            play_mode_head=self.play_mode_head,
            obs_features=self.obs_feature_bits,
            **({"logit_cap": self.logit_cap_value} if self.logit_cap_value > 0.0 else {}),
            **({"logit_cap_grad": self.logit_cap_grad} if self.logit_cap_grad != "tanh" else {}),
            **({"logit_cap_leak": self.logit_cap_leak} if self.logit_cap_leak > 0.0 else {}),
            **({"logit_cap_leak_up": self.logit_cap_leak_up} if self.logit_cap_leak_up > 0.0 else {}),
        )


    # ----------------------------------------------------- the per-entity correction

    #: Kept slot indices when `head_static` is off, as tensors registered lazily in `_encode`.
    def _dynamic_slice(self, nodes: torch.Tensor, static: Tuple[int, ...]) -> torch.Tensor:
        keep = [i for i in range(nodes.shape[-1]) if i not in static]
        return nodes[..., keep]

    def _cap_logits(self, logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        c = self.logit_cap_value
        if c <= 0.0:
            return logits
        z = logits.float()
        if mask is not None:
            legal = mask.bool() if mask.dtype != torch.bool else mask
            top = z.masked_fill(~legal, float("-inf")).amax(dim=-1, keepdim=True)
            top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))   # no legal move
        else:
            top = z.amax(dim=-1, keepdim=True)
        capped = top + c * torch.tanh((z - top) / c)
        if self.logit_cap_grad == "straight":
            capped = z + (capped - z).detach()
        elif self.logit_cap_leak > 0.0:
            capped = capped + self.logit_cap_leak * (z - z.detach())
        if self.logit_cap_grad != "straight" and self.logit_cap_leak_up > 0.0:
            capped = capped + _LeakUp.apply(z, self.logit_cap_leak_up)
        return capped.to(logits.dtype)

    def _policy_logits(self, h: torch.Tensor,
                       tokens: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        """The per-entity logits (`_entity_policy_logits`), plus the branch head's correction on the
        branch block when the network has one."""
        logits = self._entity_policy_logits(h, tokens)
        if not (self.branch_head or self.play_mode_head):
            return logits
        assert tokens is not None
        card_nodes = tokens[3]
        x = torch.cat([h, (card_nodes[..., ACTIVE_CARD_SLOT] > 0.95).to(h.dtype)], dim=-1)   # (B, H + 110)
        zero = logits.new_zeros(())
        pm = (self.play_mode_head_net(x).to(logits.dtype) if self.play_mode_head
              else zero.expand(logits.shape[0], PLAY_MODE_BLOCK))
        br = (self.branch_head_net(x).to(logits.dtype) if self.branch_head
              else zero.expand(logits.shape[0], BRANCH_BLOCK))
        a, b = ActionEncoder.PLAY_MODE_OFFSET, ActionEncoder.BRANCH_OFFSET
        return torch.cat([logits[:, :a], logits[:, a:a + PLAY_MODE_BLOCK] + pm,
                          logits[:, a + PLAY_MODE_BLOCK:b], logits[:, b:] + br], dim=-1)

    def _entity_policy_logits(self, h: torch.Tensor,
                              tokens: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        """The dense logits plus a per-entity correction, honouring the M2-family axes.

        The inherited version assumes both heads exist and that a trunk context is always fed.
        Here `head_entities` may switch one head off and `head_context` may remove the only path
        from the trunk into the correction -- so the concatenation is built rather than fixed.
        """
        base = self.policy_head(h)
        if not self.per_entity_heads or tokens is None:
            return base

        h_board, board_nodes, h_cards, card_nodes = tokens
        if not self.head_static:
            board_nodes = self._dynamic_slice(board_nodes, STATIC_BOARD_SLOTS)
            card_nodes = self._dynamic_slice(card_nodes, STATIC_CARD_SLOTS)

        # The trunk context, when there is one, is the last input block of each head's first
        # layer and the same for every entity of a sample; `_entity_out` adds its term once per
        # sample rather than concatenating it onto every entity.
        ctx: torch.Tensor | None = None
        if self.head_context:
            assert self.pe_trunk is not None
            ctx = self.pe_trunk(h)

        _A = ActionEncoder
        card_block = base[:, :_A.PLAY_MODE_OFFSET]
        gap_mid = base[:, _A.PLAY_MODE_OFFSET:_A.NODE_OFFSET] * 0.0
        country_block = base[:, _A.NODE_OFFSET:_A.BRANCH_OFFSET]
        gap_end = base[:, _A.BRANCH_OFFSET:] * 0.0

        card_corr = (self._entity_out(self.pe_card, torch.cat([h_cards, card_nodes], dim=-1), ctx)
                     if self.pe_card is not None else card_block * 0.0)
        country_corr = (self._entity_out(self.pe_country, torch.cat([h_board, board_nodes], dim=-1), ctx)
                        if self.pe_country is not None else country_block * 0.0)
        return base + torch.cat([card_corr, gap_mid, country_corr, gap_end], dim=-1)

    # ----------------------------------------------------------------- encoder

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None,
                                                  tuple[torch.Tensor, ...] | None]:
        if obs.shape[-1] != self.TOTAL_OBS_SIZE:
            raise ValueError(
                f"observation is {obs.shape[-1]} floats wide; this model reads "
                f"{self.TOTAL_OBS_SIZE}. Every slice below is taken at a fixed offset, so a "
                f"mismatched vector is misread rather than rejected.")
        b = obs.shape[0]
        attn_weights: torch.Tensor | None = None
        tokens: tuple[torch.Tensor, ...] | None = None

        if self.input_mode == "flat":
            h = self.fusion_in(self.lad_in(
                torch.index_select(obs, 1, cast(torch.Tensor, self.keep_idx))))

        elif self.input_mode == "grouped":
            board_raw = obs[:, :self.BOARD_SIZE]
            card_raw = obs[:, self.CARD_OFFSET:self.CARD_OFFSET + self.CARD_SIZE]
            glob = obs[:, self.GLOBAL_OFFSET:self.GLOBAL_OFFSET + self.GLOBAL_SIZE]
            e_board = self.lad_board(torch.index_select(
                board_raw, 1, cast(torch.Tensor, self.board_keep_idx)))
            e_card = self.lad_card(torch.index_select(
                card_raw, 1, cast(torch.Tensor, self.card_keep_idx)))
            pre = torch.cat([e_board, e_card, self.global_proj(glob)], dim=-1)
            if self.card_lookup:
                pre = torch.cat([pre, self._card_lookup(card_raw, pre, b)], dim=-1)
            tok_country = tok_cards = None
            if self.token_layers:
                t_out = self._token_path(board_raw, card_raw, glob, b)
                tok_country, tok_cards = t_out[:, 1:85], t_out[:, 85:]
                pre = torch.cat([pre, t_out[:, 0]], dim=-1)
            h = self.fusion_in(pre)
            if self.raw_tokens:
                # The tokens are the raw slots themselves. `_policy_logits` concatenates
                # (token, raw, trunk context); with raw tokens the first two would be the same
                # vector, so an empty token slice is passed and the head is sized for it.
                b_nodes = board_raw.view(b, 84, self.board_features)
                c_nodes = card_raw.view(b, 110, self.card_features)
                if self.country_identity is not None and self.card_identity is not None:
                    tok_b = self.country_identity.weight.unsqueeze(0).expand(b, -1, -1)
                    tok_c = self.card_identity.weight.unsqueeze(0).expand(b, -1, -1)
                else:
                    tok_b = b_nodes.new_zeros(b, 84, 0)
                    tok_c = c_nodes.new_zeros(b, 110, 0)
                if tok_country is not None and tok_cards is not None:
                    tok_b = torch.cat([tok_b, tok_country], dim=-1)
                    tok_c = torch.cat([tok_c, tok_cards], dim=-1)
                tokens = (tok_b, b_nodes, tok_c, c_nodes)

        else:  # entity
            board_nodes = obs[:, :self.BOARD_SIZE].view(b, 84, self.board_features)
            card_nodes = obs[:, self.CARD_OFFSET:self.CARD_OFFSET + self.CARD_SIZE] \
                .view(b, 110, self.card_features)
            if self.country_identity is not None:
                board_nodes = torch.cat(
                    [board_nodes, self.country_identity.weight.unsqueeze(0).expand(b, -1, -1)],
                    dim=-1)
            if self.card_identity is not None:
                card_nodes = torch.cat(
                    [card_nodes, self.card_identity.weight.unsqueeze(0).expand(b, -1, -1)],
                    dim=-1)

            h_board = self.lad_board_enc(board_nodes)          # (B, 84, d)
            h_cards = self.lad_card_enc(card_nodes)            # (B, 110, d)

            if self.card_self_attention:
                sa, _ = self.lad_self_attn(h_cards, h_cards, h_cards)
                h_cards = self.lad_self_ln(h_cards + sa)

            parts = [self.lad_board(_agg(h_board, self.aggregation)),
                     self.lad_card(_agg(h_cards, self.aggregation))]
            if self.cross_attention:
                ca, attn_weights = self.lad_cross_attn(h_cards, h_board, h_board)
                h_cross = self.lad_cross_ln(h_cards + ca)
                parts.append(self.lad_cross(_agg(h_cross, self.aggregation)))

            glob = obs[:, self.GLOBAL_OFFSET:self.GLOBAL_OFFSET + self.GLOBAL_SIZE]
            parts.append(self.global_proj(glob))
            h = self.fusion_in(torch.cat(parts, dim=-1))
            tokens = (h_board, board_nodes, h_cards, card_nodes)

        for block in self.res_blocks:
            h = block(h)
        return h, attn_weights, tokens

    def extract_features(self, obs: torch.Tensor, return_attn_weights: bool = False):
        h, attn, _tokens = self._encode(obs)
        if return_attn_weights:
            return h, attn
        return h


#: Slots 0..7 of a card row are the location one-hot; 8..13 are ops, rel_side, era, one_time,
#: is_scoring and ACTIVE_CARD. The split is what makes a lookup natural: 8..13 say what a card IS
#: (key material), 0..7 say where it currently is (the value being retrieved).
CARD_LOCATION_SLOTS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)


def _mlp(in_width: int, out_width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_width, out_width), nn.LayerNorm(out_width), nn.GELU())


def _agg(tokens: torch.Tensor, how: str) -> torch.Tensor:
    """(B, N, d) -> (B, N*d) keeping position, or (B, 2d) discarding it."""
    if how == "flatten":
        return tokens.reshape(tokens.shape[0], -1)
    return torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)


def ladder_config_from_state_dict(sd: Dict[str, Any]) -> Dict[str, Any] | None:
    """Recover a rung's configuration from its weights, or None if this is not a LadderNet.

    Checkpoints in this repository are bare state dicts and the architecture is detected by
    weight name, never from a recorded config. That is the stronger contract: a saved config can
    drift from the weights it claims to describe, while shapes cannot.

    **Must be tried before the v2 detection.** `lad_cross_attn.*` contains the substring
    `cross_attn`, so a cross-attention rung matches v2's test and would be rebuilt as a
    ColdWarNetV2 -- loading most tensors, silently dropping the rest, and rating a different
    network than the one that trained.
    """
    if "lad_in.0.weight" in sd:
        input_mode = "flat"
    elif "lad_board_enc.0.weight" in sd:
        input_mode = "entity"
    elif "lad_board.0.weight" in sd:
        input_mode = "grouped"
    else:
        return None

    fusion = sd["fusion_in.0.weight"]
    hidden_dim = int(fusion.shape[0])
    blocks = {k.split(".")[1] for k in sd if k.startswith("res_blocks.")}
    ident = sd.get("country_identity.weight")
    pe = sd.get("pe_trunk.weight")

    if input_mode == "flat":
        proj = sd["lad_in.0.weight"]
        in_width = int(proj.shape[1])
        entity_dim, aggregation = 0, "flatten"
    else:
        proj = sd["lad_board.0.weight"]
        if input_mode == "grouped":
            in_width = int(proj.shape[1]) + int(sd["lad_card.0.weight"].shape[1])
            entity_dim, aggregation = 0, "flatten"
        else:
            entity_dim = int(sd["lad_board_enc.0.weight"].shape[0])
            # 84*d if position was kept, 2*d if mean+max discarded it
            aggregation = "flatten" if int(proj.shape[1]) == 84 * entity_dim else "pool"
            in_width = 0

    drop_static = False
    if input_mode in ("flat", "grouped"):
        mask = static_input_mask()
        static = int(mask.sum())
        # `flat` reads the whole observation; `grouped` reads only board+card, since the global
        # block goes through `global_proj`. Comparing a grouped width against the full-observation
        # total reported drop_static=False for a run that had it on.
        full = (int(mask.numel()) if input_mode == "flat"
                else ColdWarNetV2.BOARD_SIZE + ColdWarNetV2.CARD_SIZE)
        drop_static = (in_width == full - static) and (in_width != full)

    # The M2-family head axes, all recovered from the weights like everything else here.
    has_country = "pe_country.0.weight" in sd
    has_card = "pe_card.0.weight" in sd
    head_entities = ("both" if has_country and has_card
                     else "country" if has_country
                     else "card" if has_card else "both")
    # With no heads at all the three axes are inert, and the constructor refuses anything but
    # these, so they must be recovered as the canonical values rather than inferred from weights
    # that were never built.
    head_context = pe is not None                      # pe_trunk exists only with a context
    # P22 card lookup. Recovered from shapes like everything else: cl_key exists iff the lookup
    # is built, cl_out is square at heads*dim, and the identity width is cl_key's input minus the
    # six property slots. A saved config could disagree with the weights; shapes cannot.
    cl_key = sd.get("cl_key.weight")
    if cl_key is not None:
        card_lookup = True
        hk = int(sd["cl_out.weight"].shape[0])
        cl_ident = sd.get("cl_identity")
        card_lookup_identity_dim = int(cl_ident.shape[1]) if cl_ident is not None else 0
        n_props = int(cl_key.shape[1]) - card_lookup_identity_dim
        # heads*dim is recoverable but not their factorisation; dim is pinned by convention to
        # the value the ladder uses, and heads follow. Recorded in metadata either way.
        card_lookup_dim = 32 if hk % 32 == 0 else hk
        card_lookup_heads = hk // card_lookup_dim
        del n_props
    else:
        card_lookup = False
        card_lookup_heads = card_lookup_dim = card_lookup_identity_dim = 0

    identity_dim = int(ident.shape[1]) if ident is not None else 0
    tok_in = sd.get("tok_country_in.weight")
    token_dim = int(tok_in.shape[0]) if tok_in is not None else 0
    token_layers = len({k.split(".")[2] for k in sd if k.startswith("tok_enc.layers.")})
    per_entity_heads = (int(pe.shape[0]) if pe is not None
                        else int(sd["pe_country.0.weight"].shape[0]) if has_country
                        else int(sd["pe_card.0.weight"].shape[0]) if has_card else 0)
    # head_static from the head's input width: token + raw + context, where the raw part is the
    # full slot count or the dynamic remainder.
    head_static = True
    if not (has_country or has_card):
        head_entities, head_context, head_static = "both", True, True
    if has_country or has_card:
        tok_w = (identity_dim + token_dim) if input_mode != "entity" else entity_dim
        ctx_w = per_entity_heads if head_context else 0
        if has_country:
            raw_w = int(sd["pe_country.0.weight"].shape[1]) - tok_w - ctx_w
            head_static = raw_w == ColdWarNetV2.BOARD_FEATURES
        else:
            raw_w = int(sd["pe_card.0.weight"].shape[1]) - tok_w - ctx_w
            head_static = raw_w == ColdWarNetV2.CARD_FEATURES

    return dict(
        input_mode=input_mode,
        aggregation=aggregation,
        entity_dim=entity_dim,
        card_self_attention=any(k.startswith("lad_self_attn.") for k in sd),
        cross_attention=any(k.startswith("lad_cross_attn.") for k in sd),
        per_entity_heads=per_entity_heads,
        head_context=head_context,
        head_static=head_static,
        head_entities=head_entities,
        identity_dim=identity_dim,
        card_lookup=card_lookup,
        card_lookup_heads=card_lookup_heads,
        card_lookup_dim=card_lookup_dim,
        card_lookup_identity_dim=card_lookup_identity_dim,
        token_layers=token_layers,
        token_dim=token_dim,
        drop_static=drop_static,
        hidden_dim=hidden_dim,
        num_res_blocks=len(blocks),
        entity_proj_dim=int(proj.shape[0]),
        num_attn_heads=4,
        head_center="pe_center" in sd,
        categorical_value=any(k.startswith("value_dist_head") for k in sd),
        aux_heads=any(k.startswith("aux_own_head.") for k in sd),
        card_aux=any(k.startswith("card_aux_head.") for k in sd),
        opp_legal_aux=any(k.startswith("opp_legal_head.") for k in sd),
        branch_head=any(k.startswith("branch_head_net.") for k in sd),
        play_mode_head=any(k.startswith("play_mode_head_net.") for k in sd),
        obs_features=int(sd["obs_features"]) if "obs_features" in sd else 0,
        **({"logit_cap": float(sd["logit_cap"])} if "logit_cap" in sd else {}),
    )

def create_ladder_net(device: torch.device | str, **config: Any) -> LadderNet:
    """Build a rung. Every structural axis must be named; see `LadderNet`."""
    dev = torch.device(device) if isinstance(device, str) else device
    return LadderNet(**config).to(dev)
