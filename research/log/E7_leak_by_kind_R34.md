# Where the R34 soup loses, by kind of decision (2026-10-12): influence placement is now the largest share, card choice fell from 30 to 9 points a game

**Question (owner).** Which kinds of mistakes cost the network the most: influence placement,
the headline, which card to play, or what mode to play it in? The search-disagreement bank answered
this for `E7line_swa_4720-4800M` on 2026-10-08, from a hand-made table on the fork's
`exp/search-bank`. Ask it of the current strongest network on the leaderboard, the R34 soup
`E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M` (1650).

**Answer.** Influence placement is now the largest share, and card choice has shrunk.

* **Influence placement leaks the most: +17.8 points a game, 46% of the measured leak.**
* **Which card to play comes next (+8.9), then the play mode (+6.7).**
* **Headlines (+1.3) and the Ops mode (+0.9) are small.**
* **Against the older network:**
  * Card-choice errors are down from +30.0 to +8.9 (bootstrap P = 0.9995).
  * Play-mode errors are down from +13.5 to +6.7.
  * The headline leak is down from +4.7 to +1.3.
  * **Placement has not improved** (+13.9 then, +17.8 now; P(higher now) = 0.68).
  * The total is +39.0 against +65.1.
* **The placement figure rests on few positions.**
  * Two USSR placements carry 38% of it: Arms Race's Ops at turn 7, Algeria where Zaire was
    better, and NATO's at turn 2, Iraq where Thailand was better.
  * They stand for 70 and 82 decisions of the population, and in both Gumbel@256 agrees with
    the network.
  * Placement beats card choice in 87% of bootstrap resamples: likely, but not settled.
  * The owner's earlier verdict was that board allocation is the network's strength, so the top
    placement rows ([below](#the-placement-leak)) deserve a look before this is believed.
* **Search fixes card choice and play mode, not placement.** Gumbel k=8 @256's own move recovers
  6.2 of the 8.9 points of card choice and 5.6 of the 6.7 of play mode. It recovers only 7.9 of
  the 17.8 of placement, and neither of the two heaviest placement rows.

| decision kind | R34 soup | share | k=8 @256 recovers | older network (`E7line_swa_4720-4800M`) |
|:---|---:|---:|---:|---:|
| influence placement | **+17.8 ± 6.0** | 46% | +7.9 | +13.9 ± 5.3 |
| action-round card choice | +8.9 ± 4.6 | 23% | +6.2 | **+30.0 ± 4.9** |
| play mode (event / Ops / space) | +6.7 ± 3.9 | 17% | +5.6 | +13.5 ± 5.2 |
| headline | +1.3 ± 0.8 | 3% | +1.6 | +4.7 ± 1.0 |
| Ops mode (influence / coup / realign) | +0.9 ± 0.9 | 2% | +1.1 | +2.3 ± 1.3 |
| other (realignment and coup targets, branches) | +3.3 ± 2.8 | 8% | +0.4 | +0.7 ± 0.3 |
| **all** | **+39.0 ± 9.3** | | +22.9 | +65.1 ± 8.8 |

**Units.** Points of win probability a game, both sides' decisions summed. Each figure is what
the network would gain by playing the best move at the decisions where search finds a large
error, 7% of its decisions. The other 93% show no leak a playout can detect (−17 ± 15).

**Errors.** The ± is a stratified bootstrap over positions (bank stratum × playout class, 2,000
resamples), with the playout noise included. The tables in [leak.md](E7_leak_by_kind_R34/leak.md)
print the playout noise alone, which is 2–3× smaller and understates the uncertainty.
"Other" is almost all one position: a realignment target at turn 8–10, +2.9.

## Inside each kind

* **Influence placement** (by era: turns 1–3 +4.8, turns 4–7 +6.0, turns 8–10 +7.0; USSR +10.8,
  US +7.0).
  * The turn-10 rows are final-scoring arithmetic: Sudan for Afghanistan at turn 10 is +50 points
    in every pair, and Turkey for Jordan is +36.
  * The two heaviest rows are mid-game USSR placements ([below](#the-placement-leak)).
  * By region, the largest transitions keep the region: Africa for Africa (+4.2) and Europe for
    Europe (+2.7).
* **Card choice.**
  * The cards the network plays wrongly: The China Card (+2.5) and Central America Scoring
    (+1.4).
  * The cards the best move plays instead are mostly the opponent's, spent at the right time:
    Fidel (+1.6), Muslim Revolution (+1.4), Ask Not (+1.1), Marshall Plan (+1.0).
  * As before, the leak is at DEFCON 2 (+5.8 of the 8.9).
* **Play mode.**
  * A neutral card played for influence where its event was better (+2.3; own cards +0.8).
  * An opponent's card evented where a coup was better (+1.6).
  * Influence where a realignment was better (+1.5).
  * Space where the opponent's event should have fired (+1.2).
  * By card: One Small Step (+1.9), NATO (+1.8), CIA Created (+1.3).
* **Headlines and the Ops mode.** No transition is worth more than half a point.

All the tables are in [leak.md](E7_leak_by_kind_R34/leak.md): by side, era, DEFCON, card, region
and transition, with the 20 positions that cost the most a game.

## The placement leak

The ten placement positions that cost the most a game. "Stands for" is how many decisions of the
population the row represents. "Best" is the reference search's move (Gumbel k=16 @1024 over three
seeds, valued by 256-simulation search in 64 shared worlds); k=8 @256 is the deployed searcher's
move. Gains are paired playouts, 1,024 pairs, the R34 soup greedy on both sides.

| turn/AR | side | DEFCON | Ops of | network | best (k=16 @1024) | k=8 @256 | gain a decision | stands for | a game | position |
|:---|:---|---:|:---|:---|:---|:---|---:|---:|---:|:---|
| T7 AR4 | USSR | 2 | Arms Race | Algeria | Zaire | Algeria | +9.7 ± 1.5 | 70 | +3.62 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV-2K4yAUfRd_ZyAakzR9lWUQJ7GtrFVR05kyzLvvVdM0aftjloVdFkpI6z3e73Nb9BPxPkijmTOjHtCWFqjnbmDK9DziHm1_lEUzPxt42-nJSDsjFyl_0oXVRZfAKuJ4smpWOu38ebUh09tO33TSwGmXznHKbz8X7yRbvULBB6k5S2UfjBqEQ1u8Qq3iZ_6mRMbDB_OB9z-hM5-IK2XemeBOnQE1Fm3LAg2ilz42NRpGdy94AYazBU91gcRJ6MD2jusgBmasT8YZhQh7keQj_2BWONYDQcGdE6bNIDLA3mTIHOVajGVH2Exa4DFmIDXaVgWyQg9S7yFOKgqwbgl6duJqFCtVZ5RKnpw4gpsI-SBinjhi3qhTxLK7qAb6DObJuKsYuNuLkGSoyQUm9Q7i6F6wWEXMvetqSuiG4qqlm67pNjWwWuKmxFCQH62FUP6SN0v9Sf6CPMbwb9DA_oC2pIaenqDHsZsX37ErX8V3efrXNJUPaCofslT-dZZy4t-no_xTMso7Msr_mYzyScaTjCcZTzKeZPw2Ga-LYxcbhA1Ztx-dix21B-6hPhKZ2PVGL5ZscMZa8BcMI8loZ1wPckpEDnfQglaAjxzyRMEzz0-CnQgq0EHwQUkt2E46H6YWNdXdhnnX05Fi3vACMhoux6_7jcnkZWWTp2WpPfpHUUfv3dW14pDCIMWV6YRALx6gWfpEy55MM5nEdLKC2cN5as2QO6kFhBAq8Hkm8Lwi82pqph_7HmYAbXdceQHM55_KV5xJ5yVMHxApdjvRx6HHtClJRzGp6g1oAL9xTJbHaZdGlgdw0nWEtA1p2rrCNcGUli34Hx3MASygW_NgwjzRol5dBzC85OaKUM07t5rXS0Q3YQQuINf95k4Tz1L8jteXe590lmMmdczwNSV-lCr_eZAkesvhxxVcunwkCChfFnebMF1I1VTWcr9alE4WjWhXLammNOtZ76pJFunjG6t19It-NRUHqa_LA-C2wJOMfzlnZo3UcSxeqq9fb827VA&model=off) |
| T2 AR2 | USSR | 2 | NATO | Iraq | Thailand | Iraq | +7.3 ± 2.0 | 82 | +3.20 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV_9u2yAQfhf-diUbO4mTV5kiRM05QSOAAKeNqr77DnAdZ0m1qJOmaYtQYu6D-_mdLXgjvAvSaObMoAXZ0IJ03AmmTMcj7snmW1lUxbKgabQ4W6XR4CyiFc6itEjyKv1X-KzG9WUaDcrlXzGqMZtR3mLCe6k5S2nvjRLgyKa6QK3iJ_6sIOPhlfnAu-9YmTfClTIvDLhTJ0SNJZuyIAI66WNRo2I091TNwHCyaGlREDiCDmznuA4gmLE-KWcUPewgyQf-yiw41iFBwZ0Spo2ADLBnGTJHORdj2QEX0y60GCOQmmzqgljQQuod-klJIdXVHPTsyNUAF1udUSpZcnBAMxHyAWKcVcS8UceIZXNxG-5n2E_GncXA3Q5CkjEnF5jUPfrRHbCYRYx9XdbLZU2bVd22bVHRFvPwg7XowX-Ey1JZkpkgD9HrM9at26dQjljYWMIPg7EU78W95FxzU_5Rbsob3JQ3qSnvoaa8piYy_UVucuT3s1H-LhvlP8_G11-UBxkPMh5kPMj4T8jYzg5bTIANeW83OBcravfcQzqxCug7o2dTJpyxFu0Fw2hS6o3rUE6BSHEFzWhF-MAxThI88_wI7EhJQfbAhZIaWC-dD2OJ6upqwbzo8UwxLXjAiMSosrpeGFWeLnRyt9QzaPC3vA7eu7NpxTEEIeHMdEKwFjfQLL2ReU3GnkwiXeAZFXuvyl1rRK6kBnQBKvCpJ6ppRqfZWEw_dB32ANn0XHlA5vOr8h570nmJ3YdEQt9DF5ueLlpcQFpjd8zPzi51Kg-o264XtGzWTd2ulxjhsi7XaHZwmX4s0tSP2EYUbyXnk3998w5Sj7eC_KOzFTpJdNrRTHeIe24an0sXY5sCP0iVvxlNEr3l-E4Fl24aVYSQ6XlyP5ueh1-PadWzHfSm85jSfKW-uevX41O97Rj6lN5HLhcJpm9A_NKcmDVSx24o338A-oq1lw&model=off) |
| T2 AR3 | US | 3 | Olympic Games | France | Canada | Canada | +5.1 ± 1.8 | 36 | +0.97 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV-GO2yAMfhd-56SQpL2mrzJViAbSolFAQHpXne7dZ0yaS9fqVN2madoq1AZ_YGP7cyLzRngXlTXM28EIsq4L0nEvmLYdT3gg629lsZzGqqgKWjwXDf4oIBTlNCqQKnguRzT9J6zBf1qUf8VYokfPZ3kDAe-V4QzD3lstpCfrJ3oBO81PfKslWSc8vrIQefcdUvNGuNb2hUnu9QlQ68i6LIiQnQopq0kx2aMzLJ4cGFoURB6liWznuYlSMOsC6mYUDthJlA_8lTnpWQcERX9CzFghM8C2KmaOcizWsQMs4i6wmBxQhqyrgjhphDI7OAdjAqycg4EduR7kxVZvtUZLXh7ATIJClMlPmrBg9TFh2VzaBvsZ1JP1H2LkficjyhCTj0yZHs4xnWQpiuR7W1V1U60ope2qfW7qkkLtNBBMGJyDY8LZZ4a5QVtRHdLRW0hetwenF5DPI-Q3ZfJsN2XkvbiXoqdrjso_ylF5g6PbFJX3UFReU9R-maLs-P18lL_KRvnPk_H19-VBxoOMBxkPMv4TMjazlosJ6WLe2w3ep4y6PQ-5bRGy76zBHjZPmfDWObAXLatQqbe-AxkdUeIKmtEK8IGDnyQGFvhRsmNFCrKXXGhlJOuVD_HcSDVXC_bFnHuKaSVIcEmMOrS9XhmVLnRyudQzaAi3TAwh-Jk7moMTQskPrhGBbNxAs_RG5lkZqxJFbK6g-miuWytyLo2EI6SOfKoKOs2qaTamMwxdB1VA1j3XQQL3-WV5T1Xpg4L6Aypl38sulf1qsWrbulmWqRkFflOZzFppjxXLY-rH6bKp67am1aqhVblME7A--Nz0QrKmwoR6qqH3z31_uo00KFV4V7l1U6C4-tldgoKN_LzndkM_kS7GBh0_KJ0_HjWKwXF4uaLHiweGBozPg_vZ9AJDzEgNIz8_dlw6UI2h0mlWXmj8trEZXZ_CO8dyESB-DNIn58ScVSYVxVP7_gMV7bkA&model=off) |
| T3 AR3 | US | 2 | Olympic Games | India | Pakistan | Mexico | +13.7 ± 2.1 | 13 | +0.93 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV_9u2yAQfhf-diUb7Pzwq0wRogYnaAQQ4LRR1XffAbbrKNbUddI0bRFy7PsOjrv7zg73hlgXpNHUmUFz1JICdcxxqkzHIu5R-60smqKC0RSbYge_uNimQRKyTbr4O-HVfN8kHR615V8xqtHfUT5AwCepGU1hn4ziwqH2qbqBrWJX9qwEaiMeXqkPrPsOqXlDTCnzQgVz6gqosagtC8RFJ33MalwY7VULLFwtGGoKJC5CB3p0TAfBqbE-rc0obHAUST6zV2qFox0QFNw1YdpwkQH6LEPmKMdiLD2DMs0Ci9EBqVGLC2SF5lIfYZ8UE2DlEvT0wtQgbqY6o1Sy5MQZzETIBxH9xBHzRl0ils3FaTCfQj0Z9yEG5o4iJBlicoFK3cM-uhM0RhF932NMarwje7Kr9-WG7KomxeIHa2EXP7lMU2qSqSDPcednyF13AncaSOcF0hsTOZmNCXkvvs5Q-UcZKlcYWieo_AxB5T1BGH-ZofIX-Sh_l43yn2fj66_Lg4wHGQ8yHmT8J2QcFicuyoUNeW43OBczak_M51MLF31n9OKRcmesBXvBUJwW9cZ1ICdHJL-DFrQCfGbgJwqeenYR9IJRgU6CcSW1oL10Pkz_q9s7hXnR45liVngBHvFxCRxw7jTjmqebRblcyAIa_Nq2g_duYVsxcIJL8cF1QiAbK2iW3tAyK2NVJjEdrqD6qly3hudcagFbCBXYXBXV_ITnpzGdfug6qALU9kx5Adznl-U9VqXzEuoPqBR9L7pY9rtmt9-TekMg8MhvLJPFOc2limUhHsfJvsbltik3TY3rZrNLa8LgdMoaJGsuTKinOjUkUyNQw8gNwVqTkK96IeObFganK-v387yfNx6fbFEOyfGzVPPHA0RvGbxcwaW-A0cIGF8Gt2aKzHeyopvCWYYV-zSycJYUFV6ZdJul--huJbxEDqPvc3xTMDcRpq9B_OZcqTVSx6og7z8Azyi45w&model=off) |
| T3 AR3 | US | 2 | Blockade | United Kingdom | Philippines | Philippines | +13.2 ± 2.1 | 13 | +0.90 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_ReeXQmwnSb5lVWFqE0StAQQ4LRR1X_fGXAcZxN1V32oVtvIsj1zgLmdsQVvRHZJOyuCG2xP1nVFOhl6YVwnEY9k_YNWDK42P5dZeqxquB9B5tMT3wuQFnC1WUO93I_5pv_E1eaY6pP-BAnvtJUip71zpleBrB_YBeyNPMpno8ga8fQqYpLdTyjNG5HGuBehZDBHQJ0na1qRXnU6YlVxIdpjMywdPRhqK6IOyiaxDdIm1QvnY15bUHCwVVnfy1fhVRAdEJTCMWPW9aoA4lmnwlHJxXmxh8E8CyxiANpm_17ZXtst-Mk5AUbnYBQHaQZ1MTU4Y7KloPZgBqGYFMbJEIvOHBAr5nAazBfQTy6c1STDVqWsQ04hCW034Md2SmAWGPuK87phdEXrRbuktF2uWNVw2iwhnzh4D57iKWyRy5PNJb1H789Qv26XkQMUGEt5Mowlea8-zxH9Uo7oDY5uU0T_hiJ6TRGj7adJKqF_IR30v6fj81_MnYw7GXcy7mR8EzKeZpsu0SufytxuCAEr6ncyQn4cmdh0zs5E0QfnPdhLTvC8aONCB3oORPdX0IxWgPcS4iQpiigPShw4qchOyd5oq8RGh5jGEtXsasC92HFTMQ1EBRH14xLeXI-Max4uFpV2qWfQEG-5HWIMM9tGQhC9VmeuMwLVuIEW7Y3MqzJ2ZVZ5CxsH6D5W-tb1pZZWgQtlkpy6gk0Sn6SxnHHoOugCst5IExVwXz6Wd-zKEDX0H1CpNhvVYdvzdgEDQCz2x2yHFnKryoR8tw2rF5Q3C0ZBWC2wHGkINlcLijQ1JPQRr5rZGaDJ-_-PTyR8fLMstTOZ5buZxv985mA3NTb5OfnCzwUC32sz_TRAjV7CR5VCPnIwhIDpeXK3jznlXY9HnfJmo0ynUXR-Do9X5_DqWYC3Q_-9VB-Vg43JQehTeu0IXCSYfwL4qzkK77TFZnhg7P0XLkW3xQ&model=off) |
| T3 AR5 | US | 2 | The Cambridge Five | Libya | Colombia | Libya | +4.7 ± 2.0 | 36 | +0.90 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV22L4yAQ_i9-zoIxafryV44ibjRtOKuiprtl2f9-45im6bYcyy4cx12RhszjOG_PmOobEW3sreHeDkaSzaIgrfCSa9uKhAey-UGLpijHscLnslgAtiwY_NIo8dng8zxTItKMaxNG_4pRjjGN8hYS3vdGcEx7b7VUnmzKK9RpcRLPWmU8vvIQRfsTKvNGhNb2hSvh9QlQ68iGFkSqtg-pqGnhaG7C4skprLM6KhP5zgsTleTWBVybUXCwUygfxCt3yvMW-In-hJixUmWAP_cxU5RTsY4fYBK1wGIKoDdkwwrilJG92YEfzAliovUcDfwo9KCudL3VGk15dQA7CQpRpUBZwoLVx4Rle0kN9Dn0k_UXMQq_UxFlSMpH3psO_JhW8ZRGCn7NWLVo1qxirFlWZb1oqqKpIZswOAduwjlojsVBW7E_JNfPUL12jwU-QnlTIc9WU0Hei88y9HRLEf2jFNE7FNG7BNHPEERvCULCv0hRDv3zfNDv8kH_eTq-vl8eZDzIeJDxIOM_IWM7O3NxqVzMuu3gfaqo24uQjy1Sda01s1cuvXUO7EXLGS7qrG9BxkB6eQPNaAX4ICBOEgMP4qj4kZGC7JWQujeKd70PcSwRq28m7IsZz33TRFAQkTz_F99OjEuerqZyt1QzaAj3vA4h-ItpLSAE2asL04hALe6gWXoj85qMPYkiW8BZFXqvzF1rZa6kUeBC6SimniinNza9jcUMQ9tCD5BNJ3RQwHzeKu-pJ33oofuASNV1qk1Nv1quG7quKBwI6zWoAMGpT2aHaY8tK2I6ka_Yulk2tKlo1ZR1WVWr1FqDN1g3KNfUmdBQNVxPLleBGgaFq8r1BaGaXRUoatBJi404G2fLcb5E6aOl26vHXPqN9hYDP_Q6fz0YisEJ2F3R49UDIeB8ntx9R9lNg4Gmu081hVt9CKGeUp_jH7W-PbZj6NfpAXCVIH4N0jfnxJ3tTeqLp-b9F7kjuUM&model=off) |
| T5 AR7 | US | 2 | South African Unrest | Italy | Panama | Italy | +2.0 ± 1.8 | 70 | +0.74 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV22P2yAM_i98ziQgb23_ynRCXKBtNAoISO-q0_33GUjTpO2mbjvtUxWF4ifG2H5MZT4Q70JvNHNm0AJt2gJ13AmmTMcj7tHmOy6agsAbn1UaW3jiSNKsnSHz33Zcc9Yk00jGldWkQUbtdiGR6VlKZNS5zPHDz9w-yC8Q8L7XnKWw90YJ6dDmG1nAVvETf1USbSIe3pkPvPsBqflAXCnzxiR36gSosWiDCyRk1_uY1bgw2iMzLJwsGKoLJI9SB7ZzXAcpmLE-rc0obLCTST7wd2alYx0QFNwpYdoImQH22ofMUY7FWHaAj0kLLEYHeo02tEBWatHrHeyTYgIPyjno2ZGrQS5UnVEqWXLyAGYi5IOMftKIeaOOEcvmohroM6gn4y5i4G4nQ5IhJhdYr7ewj-4ki1FE39e0pFVV43JdwrvGTVMWtCnLhkBAfrAWtvJnv1nKT7IX-kPc_hUS2O1Tjo-Q4ZjLs-WYk8_iUZK-_S1L5ItYwndYwndJwo-QRL6SpOz442zgf2UD35CB_-uR-WIy8JOMJxlPMp5kPMn4YzJeZk0XE9KGrNsNzsWM2j33uXERctsZPZsy4Yy1YC8YRtOirXEdyMmRXtxAM1oBPnDwEwXPPD9KdqSoQHvJheq1ZNve-TCmqL3BzZs-txTTFy_BIzEuadrbL-OixZpcLeUMGvw9E4P37uKN4uCD6OWF6YRALu6gWfpA85yMNZlEWkMbBLVHctUakTOpJWwhVeBTTZBpRqfZmEw_dB3UANpsufISmM9H5TPWpPM9VB8QKbdb2cWiX61w1axo7AWrNagAwbFOZt20SyXLQ-y-ynZN8Zo0lOK2bkuMSSytwenUv0G6psqEgqrg0kFmlwEKI-iTqztE-cv7A00jSRr5_lGNcnVX8xqjC6kcrS21qnRwwPNDr_LfR5VEbzkcr-DS5SNBQPo8uuvN6mj8vGO-8VCwXk57Xsd5joEs_Dyj97wlM6T-zb2Ljm8dx5fR9ym-cgQWEaaOPv7rnJg1vU6V8fkTpra7Eg&model=off) |
| T10 AR5 | US | 2 | Latin American Death Squads | Afghanistan | Sudan | Sudan | +50.0 ± 0.0 | 3 | +0.73 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV9uO2yAQ_Rc_uxL4GvtXqhVibZJYJYAAZzda7b93uJiQJg-rVupDG1mJh8Pcz9jCHwWd7CIF0XIVczG2ZTFRPRMuJ-pwU4zfUdnBhf3_Dn59uoKM0z1H3b31Np2Xc-m67pLvq-WmEaQ2w-rk2dm0SbrmdNV2eB2z65Lmtt72uxco-LgISnzZR8lnposR36CK0wt95awYEeD2nRhLpx_QmY-Cci7fCKOaXwCVyqvMbFqMa6ozjO4SZi-K-T6zMxOWHDQVls1EKuNtAwoBDiHcib4TxTSZgB-rLx4TcmYBIK-LDRTBBaVIRU6w6bXAo0tgEcVYlYViYl7EAeL4moqxG3LQkDPlK7tR1ZJz70mzE7hxkLHM5YkdZiQ_Oyy4c2qgT2CcpL4uLdUHZv0aatKWLGIPccTEiKvC5Y4xqpphwAPq2rpBbVvXuGxw0w5QkFmVglBmy5v4_nh_djm58K_QwOkIibfQ0zP02HVz8-268ll-laZv9zyhv8oTesATekgT-gpN6J6m_rdpCol_nQ_0p2ygf56M339mnmQ8yXiS8STjPyHjJTt2kZkpG3SnVWvXUXWkJhxdZrafpMhEMmupFPizklTeaC_1BGufyDLfQRmtAJ8o5FlYQww9M3KuirI4MjrzRTCyX7SxsUXD7m5DvontTJF2DIOU5mhTdfc70ejGJoxLnUGreeRiNUZn6XAKScwLu3LtEejGAzSsPoq8K3Eq_dIfrmD6cJhbOYdeCgYhGLc0TQVOUpWk2E6zThNMQTHuKTcMuA8Py6ebSm0WmD-gku33bHJjX-EKjoRds6sRdMMx7AYlO1BrP7PUgo-u2-36vu4HhPph17VDD92yq4ZBwBAY2pVGEyaqgaO_f6bgA8B92jTwX4OM_T3s1VEDlVWSnFyB1hBta2-BAWvifvOLfu4haAcfVfQA13VjC5scvPjMTwsP74_KL42i8HxZ7b8_egcB6Xl1j2K3cdXHuqoskKu4SilARjht5qXUWUtQVG9uVkG_K_PwuErmWx6Nv9exPEg-FVhH4KbE2r0R3HvnQpRchJuM9vMnUvi85Q&model=off) |
| T10 AR2 | US | 2 | NORAD | Algeria | India | Algeria | +5.7 ± 1.2 | 18 | +0.55 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_ReevRL4nvzKqkLUJglaAhbgtFHVf9_hEkKaPFRdaR92K-QEzgxzO2ML3hCbnNCKGr2qGW3rCk3MzFTqiXncou1PXJGqT2OEZ8ijLuZDkF9-L-s-o6Rqg_51PRTSuKfEooU2YMMN7u189EJg1CC59et99kFywetgM_57S8MTJHwQitGQ9kHLmRu0JTfoItmZPUuOthhw90qtY9MvqMwbYlLqF8qZkWdA9RJUZj4J64vqNyZzGXPnBQx1FeInrhzdG6Ycn6lebNgbUXCwj-6O7JUu3NAJ-HHmHDClZx4B-ixcpAgGpKIXegRh0AKLPgCh0Lap0MLVLNQe_IScICbcl6ilJyZXfqNrtJTBlOFHsOMh67gPlHjMannyWLTn1UCfQj9pc106ZvbchTUkZRwVagd-1MSpT8MHv2n6oR_bZoSnHcZxaDZV23Qd7iAjuy4LuLKXwGkoULDnxNG7f4YKTgdo3Q6KeoIi-3JebPuyvFdf5wn_VZ7wA57wQ5bwZ1jC9ywR3H2Zpxj65wnBf0oH_ufp-PpL803GNxnfZHyT8Z-Q8VQcvOjMFxd1p9UYX9HlwCwPp9eZ7yatiimdjV4WsOc0rcOmnTYTrEMgYr6DCloBPjKIEzlLLTtxeqpRhQ6czVIoTnfCWJdKtNncCfSL8nZ-kEJiOYQ0pz1Dey9Jm272xHZpCmi1j0ys1poiHMkgiFnwK9cBgWo8QOPqDZVVSV0ZluF0Bd1HYt_qOdZScXDBpWO5K0ie1XmWymnXaYIuQNsdk5YD9_FlefddaayA_gMq-W7HJ9_2cBYkdTe2dQtceoJ9nxQnNRNaljkwQfrNOLQj6ZueDGNXYwJytxrlDzwVgmrlzoSGauDcj_Po4C5wXRG44ZQynKQkzJr860ebpHWyF-1uiv2X0RdzuNGQ7KzJxtus2ITHKz2F0I9C5u8HLO3C4P1yJtxAWg8B6WV6t65JsFnnQIi_DXnHTVbx8zrl26Rw4sY6r3DWH5PmNeiLI5z02yLf7iaYq89Nyg5iz_nVCfiY4Un4z86ZLloo3xg_yPtvOYS9Rw&model=off) |
| T10 AR6 | US | 2 | Aldrich Ames Remix | Jordan | Turkey | Turkey | +35.6 ± 0.7 | 3 | +0.52 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV8uO2yAU_RevXQkw2HF-pRohxiaJVQIW4Eyj0fx7Lw8TR8liFu0sqpFFAof7PtcWvFdi8JPR3JpFj9W-ratB2JErM4iAu2r_E9VteXYwuruBYaQHPyBBA2fNbvPblj2c5zjL3_aTxabMV39diYXGJ3lY9XHUwhupFBHJ3nCOp0TzAgmfJi14TPtk1Chttcd36KzEVbwqWe0R4P43d14Mv6Ay75VQyrxxKay6AmrmKDLKYXKhqEExmyuYv85giNWVvEjt-dEK7eXIzeyibkLBwTG5O4vffJaWD8CPt9eIaTPKBPDXySeK4IFUzMzPsBmlwGIIYNLVvqmrWepx0kfwE3Oq9v1uCzp-EWqRd6LWKBUtWXkGMwFyXoY4ccCcUZeAJXNBDOQ5tJOxt6UX9ih9XENO1vNJH8CPHiQPWYTY-x0mFDOK0Y4gTCklXU0JpT3k45Z5Bk9uDZvH8kRzfjoH769Qv-EUkQsUOJRyNRxK8lF_lqMfnyUJ_yOS6DOS0FOS6GdIQn-TJEyahmCG2p4RQpue4bYHllrAvpYm9MAS-tJXCT1h6TlJ6MtJSoF_k_FNxjcZ32T892S8bA5jfJSzT7LDYm2o6HwSDvIjgYnDYPRmykdr5hnsecNJVDoYO8A6BjKND9CGVoDPAuKsvONOXCS_kKquTlKMatKSHybrfC4R7R42zJteDxtlx0kIacw6zeNG1rlTSd2yFV7cEwuLc3YTjBIQwjjJG9MRgVo8QdPqvdrWJPdkXBIG5yPoPZy61oypklqCC6m8KD2By4yUWS6mW4YBeqDaH4RyEphPr8pH6EnrJug-IFIeDnIITQ9HQ9b3rCNd2zVQjsBwaJTNMdvGnhUerGCGGaNt28LBsmVNj3YUzk1-sdAJGHxDwUpvQkuxmsErhfNAcLOAWQMXh7AM_-juIXHgPE9PE1d9NBGuHDQbCv9dnONs7d7OTQpl92F1k2xh7FaNlxj5eVLpA0Lj0s0CXjBv47UkQsD7NjtcHOIYTAgHr2BbLkfN0xjRRhmn7TX1Jv-zYntNKaVP8zrculYdXGYbwyQaa3J-EH3JkGXgLscmfBPCl-fKZzPp0B344w8Rr8Gm&model=off) |

The first two make 6.8 of the 17.8 points. Neither is a final-scoring calculation, and the
deployed search agrees with the network in both, so the evidence that the network is wrong there
is the reference search and the playouts alone.

## How this compares with the blunder census

The census and bank (`research/log/E7_blunder_census_R34.md`, branch `exp/blunder-census-r34`)
found the tail on the same network: 31 positions in 200 games whose cost holds, or is likely,
when the continuation searches. By decision kind:

| kind | rows | points |
|:---|---:|---:|
| card choice | 12 | 4.4 |
| influence placement | 10 | 2.7 |
| headline | 5 | 1.3 |
| play mode | 4 | 1.2 |

Points are the searched confirm's cost (Gumbel k=8 @256 continuation), summed over the 200 games;
placement counts the two placements of headline events.

That is the worst few positions. This entry is the population-weighted average over every
decision. Only the shares compare. Both put card choice and placement on top, and turn-10
placement is arithmetic in both. The tail gives headlines a larger share (13% against 3%): a
headline mistake is rare but can be large.

## Method

Tool `tools/search_bank.py` (`tools/README.md` §7e), ported from the fork's `exp/search-bank`
onto the current engine (`b670b258`). Branch `exp/leak-r34`, commits `68f75ca` and `870a436`.
The hand-made table of 2026-10-08 is now two stages, both checked to reproduce that run's numbers
exactly:

* `playout-select`: the playout sample;
* `leak`: the tables.

Steps, run on the fork's CI:

1. **Positions** (`annotate`, CI `38293066007`). 1,500 games of the R34 soup against itself at
   temperature 0.1, keeping one decision in 8 with two or more legal moves: 86,241 positions (460
   such decisions a game). Each was put to the network and to Gumbel k=4 @16, k=4 @64 and
   k=8 @256.
2. **Bank** (`select`, local). 8,000 positions in 72 strata: 6,000 where a searcher departs from
   the network and 2,000 agreement controls, each weighted back to its stratum's population.
   Release `search-bank-r34-20261012`, sha256 `cdaca08e…`.
3. **Reference** (`reference`, CI `38293830636`). Gumbel k=16 @1024 over three seeds, and every
   relevant move valued by 256-simulation search in 64 shared redealt worlds. The best move is
   chosen on half the worlds and the regret measured on the other half.
4. **Playout sample** (`playout-select`). 2,381 positions: all 1,310 with a search-valued regret
   of 2+ points, 771 of the other 5,390 and 300 of the 1,300 agreement controls. Release
   `playout-input-r34-20261012`, sha256 `f9549b16…`.
5. **Playouts** (`playouts`, CI `38297119333`). The network's move, the best move and k=8 @256's
   move, 1,024 pairs each, the R34 soup greedy on both sides. Each pair redeals the mover's unseen
   cards, and all moves of a pair share the dice.
6. **Tables** (`leak`, local).

The older network's column is the 2026-10-08 data (2,304 pairs, the same continuation rule), run
through the same `leak` stage and the same bootstrap.

## What this does not say

* **The best move is the reference search's.** The leak is what the network loses where a larger
  search finds a better move. Errors that search shares are not counted.
* **The continuation is the network's own greedy play.** A move whose value lies in a follow-up
  the network does not find reads worse than it is. In the older run, verdicts were most stable
  across continuations for headlines (0.92) and placement (0.77), and least for play mode (0.42).
* **Sampling dominates the error for placement.** Two positions carry 38% of it. A second bank
  from new games would show whether those two are typical.
* **One network, at one point.** R39's SWA (`research/runs.md`) now beats the R34 soup 53.2% head
  to head. It is not on the leaderboard or published yet. When it is, the same run takes it with
  one `model=` flag.

## Replicate

```bash
B=exp/leak-r34; M='_models/E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M.pt'
gh workflow run search_bank.yml --ref $B -f stage=annotate -f args="--games 1500" -f model="$M"
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py select --annotated annotate.jsonl.gz \
  --size 8000 --control 2000 --out bank.jsonl.gz
gh release create search-bank-<date> bank.jsonl.gz --prerelease
gh workflow run search_bank.yml --ref $B -f stage=reference -f release=search-bank-<date> -f model="$M" -f args="--top 4"
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py playout-select --bank bank.jsonl.gz \
  --reference reference.jsonl.gz --out playout_input/bank.jsonl.gz
gh release create playout-input-<date> playout_input/bank.jsonl.gz --prerelease
gh workflow run search_bank.yml --ref $B -f stage=playouts -f release=playout-input-<date> -f model="$M" \
  -f args="--pairs 1024"
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py leak --bank bank.jsonl.gz \
  --playouts-input playout_input/bank.jsonl.gz --playouts playouts.jsonl.gz --decisions-per-game 459.95 \
  --out leak.md
```

The bootstrap standard errors in the table above are not part of `leak`. They resample positions
within each (bank stratum, playout class) cell, 2,000 times.
