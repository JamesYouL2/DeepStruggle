// The Worker's entry (wrangler.toml `main`); the logic, testable under node, is handler.ts.
import { Env, handle } from "./handler";

export default {
  fetch(req: Request, env: Env): Promise<Response> {
    return handle(req, env);
  },
};
