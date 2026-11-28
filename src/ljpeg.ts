const SOF3 = 0xc3;
const DHT = 0xc4;
const SOS = 0xda;
const DRI = 0xdd;
const LOOKAHEAD = 16;

function lookup(bytes: Uint8Array, at: number) {
  const table = new Uint16Array(1 << LOOKAHEAD);
  let symbol = at + 16;
  let code = 0;

  for (let length = 1; length <= 16; length++) {
    for (let i = 0; i < bytes[at + length - 1]; i++) {
      const from = code << (LOOKAHEAD - length);
      table.fill((length << 8) | bytes[symbol++], from, from + (1 << (LOOKAHEAD - length)));
      code++;
    }

    code <<= 1;
  }

  return { table, end: symbol };
}

export function losslessJpeg(bytes: Uint8Array, out: Uint16Array) {
  const tables: Uint16Array[] = [];
  let precision = 0;
  let rows = 0;
  let columns = 0;
  let components = 0;
  let at = 2;

  for (;;) {
    const marker = bytes[at + 1];
    const length = (bytes[at + 2] << 8) | bytes[at + 3];

    if (bytes[at] !== 0xff || at + 4 > bytes.length) throw new Error("This raw file's sensor data is damaged.");

    if (marker === SOS) break;

    if (marker === DRI && ((bytes[at + 4] << 8) | bytes[at + 5]) > 0)
      throw new Error("This raw file packs its sensor data in a way this app can't read.");

    if (marker === SOF3) {
      precision = bytes[at + 4];
      rows = (bytes[at + 5] << 8) | bytes[at + 6];
      columns = (bytes[at + 7] << 8) | bytes[at + 8];
      components = bytes[at + 9];
    }

    if (marker === DHT) {
      for (let table = at + 4; table < at + 2 + length;) {
        const found = lookup(bytes, table + 1);
        tables[bytes[table] & 15] = found.table;
        table = found.end;
      }
    }

    at += 2 + length;
  }

  const chosen = Array.from({ length: components }, (_, c) => tables[bytes[at + 6 + 2 * c] >> 4]);
  const predictor = bytes[at + 5 + 2 * components];
  const transform = bytes[at + 7 + 2 * components] & 15;
  const row = columns * components;
  at += 2 + ((bytes[at + 2] << 8) | bytes[at + 3]);

  if (!precision || chosen.some((table) => !table) || out.length < row * rows)
    throw new Error("This raw file packs its sensor data in a way this app can't read.");

  let buffer = 0;
  let held = 0;

  const fill = () => {
    while (held <= 24) {
      let byte = at < bytes.length ? bytes[at++] : 0;

      if (byte === 0xff) {
        if (bytes[at] === 0) at++;
        else {
          at--;
          byte = 0;
        }
      }

      buffer = ((buffer << 8) | byte) >>> 0;
      held += 8;
    }
  };

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      for (let c = 0; c < components; c++) {
        fill();
        const entry = chosen[c][(buffer >>> (held - LOOKAHEAD)) & 0xffff];
        const size = entry & 0xff;
        held -= entry >> 8;
        let difference = 0;

        if (size === 16) difference = 32768;
        else if (size) {
          fill();
          const read = (buffer >>> (held - size)) & ((1 << size) - 1);
          held -= size;
          difference = read < 1 << (size - 1) ? read - (1 << size) + 1 : read;
        }

        const o = y * row + x * components + c;
        let predicted: number;

        if (y === 0) predicted = x === 0 ? 1 << (precision - transform - 1) : out[o - components];
        else if (x === 0) predicted = out[o - row];
        else {
          const left = out[o - components];
          const above = out[o - row];
          const corner = out[o - row - components];

          predicted =
            predictor === 1
              ? left
              : predictor === 2
                ? above
                : predictor === 3
                  ? corner
                  : predictor === 4
                    ? left + above - corner
                    : predictor === 5
                      ? left + ((above - corner) >> 1)
                      : predictor === 6
                        ? above + ((left - corner) >> 1)
                        : (left + above) >> 1;
        }

        out[o] = predicted + difference;
      }
    }
  }

  return { rows, row };
}
