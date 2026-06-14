export type PreviewSubscriberInput = {
  email: string;
  userId?: string | null;
};

export type SubscriberStorageResult =
  | { stored: true; storage: "supabase"; row_count: number }
  | {
      stored: false;
      storage: "not_configured" | "supabase";
      error: string;
      status?: number;
    };

type SupabaseConfig = {
  url: string;
  serviceRoleKey: string;
  table: string;
};

function supabaseConfig(): SupabaseConfig | null {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL?.replace(/\/+$/, "");
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceRoleKey) return null;
  return {
    url,
    serviceRoleKey,
    table: process.env.SUPABASE_SUBSCRIBERS_TABLE || "subscribers",
  };
}

export async function ensurePreviewSubscriber(
  input: PreviewSubscriberInput,
): Promise<SubscriberStorageResult> {
  const config = supabaseConfig();
  if (!config) {
    return {
      stored: false,
      storage: "not_configured",
      error: "supabase_env_missing",
    };
  }

  const email = input.email.trim().toLowerCase();
  if (!email) {
    return {
      stored: false,
      storage: "supabase",
      error: "missing_email",
      status: 400,
    };
  }

  const now = new Date();
  const expires = new Date(now.getTime() + 5 * 24 * 60 * 60 * 1000);
  const row = {
    email,
    user_id: input.userId || null,
    tier: "free_signup",
    plan_id: "preview",
    status: "trialing",
    preview_started_at: now.toISOString(),
    preview_expires_at: expires.toISOString(),
    updated_at: now.toISOString(),
  };

  const response = await fetch(
    `${config.url}/rest/v1/${config.table}?on_conflict=email`,
    {
      method: "POST",
      headers: {
        apikey: config.serviceRoleKey,
        authorization: `Bearer ${config.serviceRoleKey}`,
        "content-type": "application/json",
        prefer: "resolution=ignore-duplicates,return=representation",
      },
      body: JSON.stringify(row),
    },
  );

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    return {
      stored: false,
      storage: "supabase",
      status: response.status,
      error: text || response.statusText || "subscriber_write_failed",
    };
  }

  let rowCount = 1;
  try {
    const payload = (await response.json()) as unknown;
    rowCount = Array.isArray(payload) ? payload.length : 1;
  } catch {
    rowCount = 1;
  }

  return { stored: true, storage: "supabase", row_count: rowCount };
}
