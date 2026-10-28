import { trainerStatus } from "@/lib/hub";

export const runtime = "nodejs";

// GET -> which trainers have their key set (names only)
export async function GET() {
  return Response.json(await trainerStatus());
}
