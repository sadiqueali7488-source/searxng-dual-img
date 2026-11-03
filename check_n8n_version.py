import subprocess

script = '''
const sqlite3 = require("/usr/local/lib/node_modules/n8n/node_modules/sqlite3");
const db = new sqlite3.Database("/home/node/.n8n/database.sqlite");
db.get("SELECT value FROM settings WHERE key = \'versionCli\'", (err, row) => {
  console.log("n8n version:", row ? row.value : "unknown");
});
'''

res = subprocess.run(['docker', 'exec', 'N8N', 'node', '-e', script], capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("ERR:", res.stderr)
