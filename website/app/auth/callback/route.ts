import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import { ensurePreviewSubscriber } from "@/lib/subscriber-access";

export async function GET(request: Request) {
  const requestUrl = new URL(request.url);
  const code = requestUrl.searchParams.get("code");

  if (!code) {
    return NextResponse.redirect(`${requestUrl.origin}/sign-in?error=missing-code`);
  }

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    return NextResponse.redirect(`${requestUrl.origin}/sign-in?error=missing-env`);
  }

  const supabase = createClient(supabaseUrl, supabaseAnonKey);
  const { data, error } = await supabase.auth.exchangeCodeForSession(code);

  if (error) {
    return NextResponse.redirect(`${requestUrl.origin}/sign-in?error=auth-callback-failed`);
  }

  const email = data.user?.email;
  if (email) {
    const storage = await ensurePreviewSubscriber({
      email,
      userId: data.user?.id ?? null,
    });
    if (!storage.stored && storage.storage === "supabase") {
      console.error("subscriber_preview_upsert_failed", storage);
    }
  }

  return NextResponse.redirect(`${requestUrl.origin}/portal`);
}
