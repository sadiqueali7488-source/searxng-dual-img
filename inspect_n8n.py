import subprocess
import json

script = '''
const sqlite3 = require("/usr/local/lib/node_modules/n8n/node_modules/sqlite3");
const db = new sqlite3.Database("/home/node/.n8n/database.sqlite");
db.all("SELECT id, name, active, nodes, connections FROM workflow_entity", (err, rows) => {
  if (err) { console.error(err); return; }
  for (const r of rows) {
    console.log("==========================================");
    console.log("WORKFLOW:", r.id, r.name, "ACTIVE:", r.active);
    const nodes = JSON.parse(r.nodes);
    console.log("NODE NAMES:", nodes.map(n => n.name));
    for (const n of nodes) {
      if (n.parameters && n.parameters.jsCode) {
        console.log("--- NODE:", n.name, "---");
        console.log(n.parameters.jsCode);
      }
    }
  }
});
'''

res = subprocess.run(['docker', 'exec', 'N8N', 'node', '-e', script], capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("ERR:", res.stderr)
