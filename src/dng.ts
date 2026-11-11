// Raw files (DNG and other TIFF-based raws) start with a tiny thumbnail that browsers decode
// instead of the picture; the camera's full-size JPEG preview sits in a sub-image.
export async function rawPreview(file: Blob): Promise<Blob | undefined> {
  const head = new DataView(await file.slice(0, 8).arrayBuffer());
  const order = head.byteLength === 8 ? head.getUint16(0) : 0;

  if (order !== 0x4949 && order !== 0x4d4d) return;
  const little = order === 0x4949;

  if (head.getUint16(2, little) !== 42) return;
  const data = new DataView(await file.arrayBuffer());
  const u16 = (o: number) => data.getUint16(o, little);
  const u32 = (o: number) => data.getUint32(o, little);
  const value = (o: number) => (u16(o + 2) === 3 ? u16(o + 8) : u32(o + 8));
  let best: { offset: number; length: number } | undefined;
  const seen = new Set<number>();
  const queue = [u32(4)];

  while (queue.length) {
    const ifd = queue.pop()!;

    if (!ifd || seen.has(ifd) || ifd + 2 > data.byteLength) continue;
    seen.add(ifd);
    const count = u16(ifd);
    const tags = new Map<number, number>();

    for (let i = 0; i < count; i++) {
      const entry = ifd + 2 + i * 12;
      const tag = u16(entry);

      if (tag === 330) {
        const n = u32(entry + 4);
        const at = n === 1 ? entry + 8 : u32(entry + 8);

        for (let k = 0; k < n; k++) queue.push(u32(n === 1 ? at : at + 4 * k));
      } else tags.set(tag, value(entry));
    }

    queue.push(u32(ifd + 2 + count * 12));
    const offset = tags.get(513) ?? tags.get(273);
    const length = tags.get(514) ?? tags.get(279);
    const jpeg = tags.get(259) === 6 || tags.get(259) === 7;

    // NewSubfileType 1 marks a preview; type 0 is the raw sensor data, which browsers can't decode.
    if (offset && length && jpeg && tags.get(254) === 1 && length > (best?.length ?? 0)) best = { offset, length };
  }

  return best && file.slice(best.offset, best.offset + best.length, "image/jpeg");
}
