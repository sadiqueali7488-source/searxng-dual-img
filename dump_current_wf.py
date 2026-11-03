import subprocess
import json

script = '''
const sqlite3 = require("/usr/local/lib/node_modules/n8n/node_modules/sqlite3");
const db = new sqlite3.Database("/home/node/.n8n/database.sqlite");
db.get("SELECT id, name, nodes, connections FROM workflow_entity WHERE id = \'jPHVv0xMEkwPLSQb\'", (err, row) => {
  if (err) { console.error(err); return; }
  console.log(JSON.stringify({
    id: row.id,
    name: row.name,
    nodes: JSON.parse(row.nodes),
    connections: JSON.parse(row.connections)
  }, null, 2));
});
'''

res = subprocess.run(['docker', 'exec', 'N8N', 'node', '-e', script], capture_output=True, text=True)
with open('c:/Users/Administrator/Desktop/image search/current_jPHV.json', 'w', encoding='utf-8') as f:
    f.write(res.stdout)
print("Saved to current_jPHV.json, length:", len(res.stdout))
