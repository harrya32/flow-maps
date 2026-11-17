// Render DOT with the bundled Graphviz WebAssembly package.
// Usage: node render_graphviz.cjs VIZ_MODULE INPUT_DOT OUTPUT_JSON
const fs = require('node:fs');
const { instance } = require(process.argv[2]);

(async () => {
  const viz = await instance();
  const dot = fs.readFileSync(process.argv[3], 'utf8');
  const result = viz.renderJSON(dot, { engine: 'dot' });
  fs.writeFileSync(process.argv[4], JSON.stringify(result, null, 2));
})().catch(error => {
  process.stderr.write(String(error) + '\n');
  process.exitCode = 1;
});
