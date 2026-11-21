type Table = Partial<Record<string, Record<string, number[]>>>;

export async function blackOffsets(body: string, iso: number) {
  const { default: table }: { default: Table } = await import("./blacks.json");
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(body)));
  const rows = table[Array.from(digest.subarray(0, 8), (byte) => byte.toString(16).padStart(2, "0")).join("")];

  const isos = Object.keys(rows ?? {})
    .map(Number)
    .toSorted((a, b) => a - b);

  const above = isos.findIndex((step) => step >= iso);

  if (!rows || !iso || above < 0 || (above === 0 && isos[0] !== iso)) return [0, 0, 0, 0];
  const low = isos[Math.max(0, above - 1)];
  const high = isos[above];
  const along = high === low ? 0 : Math.log(iso / low) / Math.log(high / low);

  return rows[low].map((offset, plane) => offset + along * (rows[high][plane] - offset));
}
