import { NextResponse } from "next/server"
import { createClient } from "@supabase/supabase-js"

export async function GET(request: Request) {
  const requestUrl = new URL(request.url)
  const code = requestUrl.searchParams.get("code")

  if (code) {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
    const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

    if (!supabaseUrl || !supabaseAnonKey) {
      return NextResponse.redirect(`${requestUrl.origin}/sign-in?error=missing-env`)
    }

    const supabase = createClient(supabaseUrl, supabaseAnonKey)
    const { error } = await supabase.auth.exchangeCodeForSession(code)

    if (error) {
      return NextResponse.redirect(`${requestUrl.origin}/sign-in?error=auth-callback-failed`)
    }
  }

  return NextResponse.redirect(`${requestUrl.origin}/portal`)
}
