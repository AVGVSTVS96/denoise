// Sign in with the user's own OpenRouter account (OAuth PKCE): the key lives only in this browser
// and every cloud clean is billed to that account.
const KEY = "openrouter-key";
const VERIFIER = "openrouter-verifier";

export const cloudKey = () => localStorage.getItem(KEY) ?? undefined;

export const signOut = () => localStorage.removeItem(KEY);

const base64url = (bytes: Uint8Array) =>
  btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");

export async function signIn() {
  const verifier = base64url(crypto.getRandomValues(new Uint8Array(32)));

  const challenge = base64url(
    new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier))),
  );

  sessionStorage.setItem(VERIFIER, verifier);
  const url = new URL("https://openrouter.ai/auth");
  url.searchParams.set("callback_url", location.origin + location.pathname);
  url.searchParams.set("code_challenge", challenge);
  url.searchParams.set("code_challenge_method", "S256");
  location.assign(url);
}

export async function finishSignIn() {
  const params = new URLSearchParams(location.search);
  const code = params.get("code");
  const verifier = sessionStorage.getItem(VERIFIER);

  if (!code || !verifier) return;
  history.replaceState(null, "", location.pathname);
  sessionStorage.removeItem(VERIFIER);

  const res = await fetch("https://openrouter.ai/api/v1/auth/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, code_verifier: verifier, code_challenge_method: "S256" }),
  });

  if (!res.ok) throw new Error(`OpenRouter sign-in failed (${res.status})`);
  const { key }: { key: string } = await res.json();
  localStorage.setItem(KEY, key);
}
