export type BriefIngestBody = {
  schema_version: string;
  brief_metadata: {
    brief_id: string;
    trading_date_ist: string;
  };
  [key: string]: unknown;
};

export type BriefValidationResult =
  | { ok: true }
  | { ok: false; error: string };

export type BriefStoreResult =
  | {
      stored: true;
      storage: "supabase";
      row_count: number;
    }
  | {
      stored: false;
      storage: "not_configured" | "supabase";
      error: string;
      status?: number;
    };

type SupabaseConfig = {
  url: string;
  serviceRoleKey: string;
};

export function validateBriefPayload(
  value: unknown,
): BriefValidationResult {
  if (!value || typeof value !== "object") {
    return { ok: false, error: "invalid_body" };
  }
  const body = value as Partial<BriefIngestBody>;
  if (!body.schema_version || typeof body.schema_version !== "string") {
    return { ok: false, error: "missing_schema_version" };
  }
  if (!body.brief_metadata || typeof body.brief_metadata !== "object") {
    return { ok: false, error: "missing_brief_metadata" };
  }
  const md = body.brief_metadata as Partial<BriefIngestBody["brief_metadata"]>;
  if (!md.brief_id || typeof md.brief_id !== "string") {
    return { ok: false, error: "missing_brief_id" };
  }
  if (!md.trading_date_ist || typeof md.trading_date_ist !== "string") {
    return { ok: false, error: "missing_trading_date_ist" };
  }
  return { ok: true };
}

function supabaseConfig(): SupabaseConfig | null {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL?.replace(/\/+$/, "");
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceRoleKey) return null;
  return { url, serviceRoleKey };
}

export async function storeBrief(
  brief: BriefIngestBody,
): Promise<BriefStoreResult> {
  const config = supabaseConfig();
  if (!config) {
    return {
      stored: false,
      storage: "not_configured",
      error: "supabase_env_missing",
    };
  }

  const row = {
    brief_id: brief.brief_metadata.brief_id,
    trading_date_ist: brief.brief_metadata.trading_date_ist,
    schema_version: brief.schema_version,
    payload: brief,
  };
  const response = await fetch(
    `${config.url}/rest/v1/briefs?on_conflict=brief_id`,
    {
      method: "POST",
      headers: {
        apikey: config.serviceRoleKey,
        authorization: `Bearer ${config.serviceRoleKey}`,
        "content-type": "application/json",
        prefer: "resolution=merge-duplicates,return=representation",
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
      error: text || response.statusText || "supabase_write_failed",
    };
  }

  let rowCount = 1;
  try {
    const payload = (await response.json()) as unknown;
    rowCount = Array.isArray(payload) ? payload.length : 1;
  } catch {
    rowCount = 1;
  }
  return {
    stored: true,
    storage: "supabase",
    row_count: rowCount,
  };
}

