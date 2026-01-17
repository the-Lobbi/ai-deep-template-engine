# 🔧 Manual Studio Connection Guide

## Current Situation

The Cloudflare tunnel is initializing. While waiting, you can manually add the origin in Studio.

## ✅ Solution: Manual Origin Configuration

Studio has a feature to manually add allowed origins. Here's how:

### Step 1: Open Studio
Go to: https://smith.langchain.com/studio/

### Step 2: Click "Connect to a local server"
You should see a button or link that says:
- "Connect to a local server"
- OR "Add server"
- OR "Connect to different server"

### Step 3: Add the Server URL

In the connection dialog, you'll see options:

#### Option A: Use Tunnel URL (Recommended)
```
https://ben-photograph-sparc-opponents.trycloudflare.com
```

#### Option B: Use Localhost (if Studio Desktop)
```
http://127.0.0.1:8123
```

### Step 4: Allow the Origin

If you see an error about "origin not allowed":

1. Look for a button or option that says:
   - "Add to allowed origins"
   - "Allow this origin"
   - "Trust this server"

2. Click it to add the origin to your allowed list

3. Try connecting again

## 🎯 Current Server Status

- **Local Server**: ✅ Running on port 8123
- **Tunnel URL**: https://ben-photograph-sparc-opponents.trycloudflare.com
- **Status**: Tunnel initializing (may take 1-2 minutes)

## 🔄 Alternative: Wait for Tunnel

The Cloudflare tunnel takes 1-2 minutes to fully initialize. You can:

1. **Wait 2 minutes**
2. **Try this URL**:
   ```
   https://smith.langchain.com/studio/?baseUrl=https://ben-photograph-sparc-opponents.trycloudflare.com
   ```
3. If it still fails, use manual connection above

## 🖥️ Best Solution: Use Desktop App

The easiest way to avoid CORS issues entirely:

### Download LangGraph Studio Desktop
1. Go to: **https://studio.langchain.com**
2. Download the desktop application
3. Install and open it
4. Use localhost directly: `http://127.0.0.1:8123`
5. **No CORS issues, no tunnel needed!**

The desktop app doesn't have browser security restrictions.

## 📝 Manual Connection Steps (Detailed)

When you see the Studio interface:

### 1. Look for Connection Settings
- Top right corner
- Settings icon
- "Server" or "Connection" menu

### 2. Connection Dialog Options

You might see:
```
┌────────────────────────────────────┐
│ Connect to Agent Server            │
├────────────────────────────────────┤
│ Server URL:                        │
│ [                                ] │
│                                    │
│ □ Add to allowed origins           │
│                                    │
│ [Cancel]            [Connect]      │
└────────────────────────────────────┘
```

### 3. Enter Server URL
Paste:
```
https://ben-photograph-sparc-opponents.trycloudflare.com
```

### 4. Check "Add to allowed origins"
If there's a checkbox, check it!

### 5. Click Connect

## 🔍 Troubleshooting

### "Origin not allowed" Error

**Solution 1: Check Tunnel Status**
```bash
curl https://ben-photograph-sparc-opponents.trycloudflare.com/ok
```

Should return: `{"ok":true}`

If you get a 530 error, wait 1-2 minutes for tunnel initialization.

**Solution 2: Use Manual Origin Add**
1. In Studio, go to Settings
2. Find "Allowed Origins" or "Trusted Servers"
3. Add the tunnel URL manually
4. Save and retry

**Solution 3: Restart with Fresh Tunnel**
```bash
# Stop server
kill $(cat langgraph.pid)

# Start with new tunnel
langgraph dev --no-browser --port 8123 --tunnel
```

This generates a NEW tunnel URL.

### "Can't reach server"

**Check local server**:
```bash
curl http://localhost:8123/ok
```

Should return: `{"ok":true}`

**If not running**:
```bash
source venv/bin/activate
langgraph dev --no-browser --port 8123 --tunnel
```

## 🎨 Once Connected

After successful connection:

1. **Select "connected_graph"** from dropdown
2. You'll see **30 nodes** fully connected
3. Click **"Run"** to execute
4. Watch workflow flow through all phases

## 📊 What You Should See

```
START
  ↓
Planning (2 nodes)
  ↓
Repo-Analyst (7 nodes)
  ↓
Design (5 nodes)
  ↓
Approval Gate (1 node)
  ↓
Resource Creation (6 nodes)
  ↓
Deployment (5 nodes)
  ↓
Finalize (1 node)
  ↓
END
```

All nodes connected in a clear flow!

## 🔐 Security Note

The tunnel URL is temporary and will change when you restart the server. Don't share it publicly as it provides access to your local dev server.

## ✅ Quick Checklist

- [ ] Server running locally (check with `curl localhost:8123/ok`)
- [ ] Tunnel URL generated (check logs)
- [ ] Wait 1-2 minutes for tunnel initialization
- [ ] Open Studio
- [ ] Click "Connect to local server"
- [ ] Enter tunnel URL
- [ ] Check "Add to allowed origins" if available
- [ ] Click Connect
- [ ] Select "connected_graph"
- [ ] Start visualizing!

## 🎯 Current Tunnel URL

**Use this URL in Studio**:
```
https://ben-photograph-sparc-opponents.trycloudflare.com
```

**Full Studio Link**:
```
https://smith.langchain.com/studio/?baseUrl=https://ben-photograph-sparc-opponents.trycloudflare.com
```

---

**Note**: If tunnel issues persist, the Desktop app is the most reliable solution and bypasses all CORS/tunnel complexity.
