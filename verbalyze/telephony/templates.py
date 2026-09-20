"""
verbalyze/telephony/templates.py

HTML and JavaScript interfaces for Telephony Supervisor Console and In-Browser Client.
Zero-emoji compliant.
"""

SUPERVISOR_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Verbalyze Telephony - Supervisor Live Console</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        body { background: #0f172a; color: #f8fafc; padding: 24px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #1e293b; }
        .title { font-size: 22px; font-weight: 700; color: #38bdf8; }
        .subtitle { font-size: 13px; color: #94a3b8; }
        .badge { font-size: 11px; padding: 4px 10px; border-radius: 9999px; font-weight: 600; text-transform: uppercase; }
        .badge-live { background: #065f46; color: #34d399; }
        .grid-kpi { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px; }
        .card-label { font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase; margin-bottom: 6px; }
        .card-value { font-size: 28px; font-weight: 700; color: #f8fafc; }
        .alert-banner { display: none; padding: 14px 18px; background: #7f1d1d; border-left: 5px solid #ef4444; color: #fecaca; border-radius: 6px; margin-bottom: 24px; font-weight: 500; font-size: 14px; }
        .table-container { background: #1e293b; border: 1px solid #334155; border-radius: 8px; overflow: hidden; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
        th { background: #0f172a; color: #94a3b8; padding: 14px 16px; font-weight: 600; border-bottom: 1px solid #334155; }
        td { padding: 14px 16px; border-bottom: 1px solid #334155; color: #e2e8f0; }
        tr:hover { background: #243248; }
        .btn { padding: 6px 12px; font-size: 12px; font-weight: 600; border-radius: 6px; border: none; cursor: pointer; transition: background 0.15s; }
        .btn-whisper { background: #0284c7; color: white; margin-right: 6px; }
        .btn-whisper:hover { background: #0369a1; }
        .btn-barge { background: #d97706; color: white; margin-right: 6px; }
        .btn-barge:hover { background: #b45309; }
        .btn-takeover { background: #dc2626; color: white; }
        .btn-takeover:hover { background: #b91c1c; }
        .tag { font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
        .tag-calm { background: #064e3b; color: #6ee7b7; }
        .tag-elevated { background: #78350f; color: #fde68a; }
        .tag-agitated { background: #7f1d1d; color: #fca5a5; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); justify-content: center; align-items: center; }
        .modal-content { background: #1e293b; border: 1px solid #475569; padding: 24px; border-radius: 8px; width: 440px; }
        .modal-title { font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #38bdf8; }
        .modal-input { width: 100%; padding: 10px; background: #0f172a; border: 1px solid #475569; color: white; border-radius: 6px; margin-bottom: 16px; font-size: 13px; }
        .modal-actions { display: flex; justify-content: flex-end; gap: 8px; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="title">Verbalyze Telephony - Supervisor Live Console</div>
            <div class="subtitle">Real-time Fleet Observability, Live Agitation Alerts & Whisper Coaching</div>
        </div>
        <div>
            <span class="badge badge-live" id="stream-status">Stream Connected</span>
        </div>
    </div>

    <div class="alert-banner" id="alert-banner"></div>

    <div class="grid-kpi">
        <div class="card">
            <div class="card-label">Active Calls</div>
            <div class="card-value" id="kpi-active">0</div>
        </div>
        <div class="card">
            <div class="card-label">High Agitation Calls</div>
            <div class="card-value" style="color: #f87171;" id="kpi-agitation">0</div>
        </div>
        <div class="card">
            <div class="card-label">Code-Switched Calls</div>
            <div class="card-value" style="color: #fbbf24;" id="kpi-codeswitch">0</div>
        </div>
        <div class="card">
            <div class="card-label">Average Jitter (ms)</div>
            <div class="card-value" style="color: #38bdf8;" id="kpi-jitter">0.0</div>
        </div>
        <div class="card">
            <div class="card-label">Average Fleet MOS</div>
            <div class="card-value" style="color: #4ade80;" id="kpi-mos">4.20</div>
        </div>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Call ID</th>
                    <th>Customer</th>
                    <th>Persona</th>
                    <th>Language</th>
                    <th>Sentiment / Agitation</th>
                    <th>Jitter</th>
                    <th>MOS</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody id="calls-tbody">
                <tr><td colspan="9" style="text-align: center; color: #94a3b8; padding: 24px;">No active calls in fleet</td></tr>
            </tbody>
        </table>
    </div>

    <div class="modal" id="whisper-modal">
        <div class="modal-content">
            <div class="modal-title" id="whisper-modal-title">Inject Whisper Coaching</div>
            <p style="font-size: 12px; color: #94a3b8; margin-bottom: 12px;">This guidance is delivered privately into the AI agent context without the customer hearing.</p>
            <input type="text" class="modal-input" id="whisper-text" placeholder="e.g. Offer 100% late fee waiver if customer pays by 5 PM" />
            <input type="hidden" id="whisper-call-id" />
            <div class="modal-actions">
                <button class="btn" style="background:#475569; color:white;" onclick="closeWhisperModal()">Cancel</button>
                <button class="btn btn-whisper" onclick="submitWhisper()">Send Whisper</button>
            </div>
        </div>
    </div>

    <script>
        let ws;
        function connectWS() {
            const loc = window.location;
            const wsUrl = (loc.protocol === 'https:' ? 'wss://' : 'ws://') + loc.host + '/telephony/supervisor/stream';
            ws = new WebSocket(wsUrl);
            ws.onopen = () => {
                document.getElementById('stream-status').textContent = 'Stream Connected';
                document.getElementById('stream-status').className = 'badge badge-live';
            };
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleEvent(data);
                } catch(e) {}
            };
            ws.onclose = () => {
                document.getElementById('stream-status').textContent = 'Disconnected (Reconnecting...)';
                document.getElementById('stream-status').className = 'badge';
                setTimeout(connectWS, 2000);
            };
        }

        function refreshCallsList() {
            fetch('/telephony/supervisor/calls')
                .then(r => r.json())
                .then(data => {
                    renderSummary(data.summary || {});
                    renderCalls(data.calls || []);
                }).catch(() => {});
        }

        function renderSummary(summary) {
            document.getElementById('kpi-active').textContent = summary.active_calls_count || 0;
            document.getElementById('kpi-agitation').textContent = summary.high_agitation_calls || 0;
            document.getElementById('kpi-codeswitch').textContent = summary.code_switched_calls || 0;
            document.getElementById('kpi-jitter').textContent = (summary.average_jitter_ms || 0).toFixed(1);
            document.getElementById('kpi-mos').textContent = (summary.average_mos_score || 4.2).toFixed(2);
        }

        function renderCalls(calls) {
            const tbody = document.getElementById('calls-tbody');
            if (!calls || calls.length === 0) {
                tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: #94a3b8; padding: 24px;">No active calls in fleet</td></tr>';
                return;
            }
            tbody.innerHTML = calls.map(c => {
                let tagClass = 'tag-calm';
                if (c.sentiment_category === 'ELEVATED') tagClass = 'tag-elevated';
                else if (c.sentiment_category === 'AGITATED' || c.sentiment_category === 'HOSTILE') tagClass = 'tag-agitated';

                return `<tr>
                    <td style="font-family: monospace; font-size: 12px;">${c.call_id}</td>
                    <td>${c.caller_phone_masked}</td>
                    <td>${c.persona}</td>
                    <td><span class="tag" style="background:#1e3a8a; color:#93c5fd;">${c.language.toUpperCase()}</span></td>
                    <td><span class="tag ${tagClass}">${c.sentiment_category} (${c.agitation_score.toFixed(2)})</span></td>
                    <td>${c.jitter_ms.toFixed(1)} ms</td>
                    <td>${c.mos_score.toFixed(2)}</td>
                    <td><span class="tag" style="background:#334155; color:#cbd5e1;">${c.status}</span></td>
                    <td>
                        <button class="btn btn-whisper" onclick="openWhisperModal('${c.call_id}')">Whisper</button>
                        <button class="btn btn-barge" onclick="triggerBargeIn('${c.call_id}')">Barge-In</button>
                        <button class="btn btn-takeover" onclick="triggerTakeover('${c.call_id}')">Takeover</button>
                    </td>
                </tr>`;
            }).join('');
        }

        function handleEvent(data) {
            if (data.alert) {
                const banner = document.getElementById('alert-banner');
                banner.style.display = 'block';
                banner.textContent = '[ALERT]: ' + data.alert + ' on Call ' + data.call_id + ' (Agitation: ' + (data.agitation_score || 0).toFixed(2) + ')';
                setTimeout(() => { banner.style.display = 'none'; }, 8000);
            }
            refreshCallsList();
        }

        function openWhisperModal(callId) {
            document.getElementById('whisper-call-id').value = callId;
            document.getElementById('whisper-modal-title').textContent = 'Inject Whisper to Call: ' + callId;
            document.getElementById('whisper-text').value = '';
            document.getElementById('whisper-modal').style.display = 'flex';
        }

        function closeWhisperModal() {
            document.getElementById('whisper-modal').style.display = 'none';
        }

        function submitWhisper() {
            const callId = document.getElementById('whisper-call-id').value;
            const text = document.getElementById('whisper-text').value.trim();
            if (!text) return;
            fetch('/telephony/supervisor/whisper', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({call_id: callId, text: text, supervisor_id: 'supervisor_web'})
            }).then(r => r.json()).then(res => {
                closeWhisperModal();
                refreshCallsList();
            });
        }

        function triggerBargeIn(callId) {
            fetch('/telephony/supervisor/barge-in', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({call_id: callId, supervisor_id: 'supervisor_web'})
            }).then(() => refreshCallsList());
        }

        function triggerTakeover(callId) {
            if (!confirm('Initiate immediate supervisor takeover for call ' + callId + '?')) return;
            fetch('/telephony/supervisor/takeover', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({call_id: callId, supervisor_id: 'supervisor_web'})
            }).then(() => refreshCallsList());
        }

        connectWS();
        refreshCallsList();
    </script>
</body>
</html>
"""

BROWSER_CLIENT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Verbalyze Telephony - In-Browser Voice Client</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        body { background: #0f172a; color: #f8fafc; padding: 24px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 720px; }
        .header { margin-bottom: 20px; text-align: center; }
        .title { font-size: 22px; font-weight: 700; color: #38bdf8; }
        .subtitle { font-size: 13px; color: #94a3b8; margin-top: 4px; }
        .panel { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
        .controls-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
        label { font-size: 12px; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 6px; }
        select { width: 100%; padding: 10px; background: #0f172a; border: 1px solid #475569; color: white; border-radius: 6px; font-size: 13px; }
        .call-btn { width: 100%; padding: 14px; font-size: 15px; font-weight: 700; border-radius: 6px; border: none; cursor: pointer; transition: background 0.2s; }
        .call-btn-start { background: #059669; color: white; }
        .call-btn-start:hover { background: #047857; }
        .call-btn-end { background: #dc2626; color: white; }
        .call-btn-end:hover { background: #b91c1c; }
        .status-badge { display: inline-block; font-size: 12px; padding: 4px 12px; border-radius: 9999px; font-weight: 600; text-transform: uppercase; margin-top: 12px; }
        .status-idle { background: #334155; color: #cbd5e1; }
        .status-live { background: #065f46; color: #34d399; }
        canvas { width: 100%; height: 60px; background: #0f172a; border-radius: 6px; margin-top: 12px; }
        .transcript-box { height: 260px; overflow-y: auto; background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 14px; display: flex; flex-direction: column; gap: 10px; font-size: 13px; }
        .msg { padding: 8px 12px; border-radius: 6px; max-width: 85%; }
        .msg-agent { background: #1e3a8a; color: #e0f2fe; align-self: flex-start; }
        .msg-user { background: #065f46; color: #d1fae5; align-self: flex-end; }
        .meta-chips { display: flex; gap: 8px; margin-top: 12px; }
        .chip { font-size: 11px; padding: 4px 8px; border-radius: 4px; background: #334155; color: #cbd5e1; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">Verbalyze Telephony - In-Browser Voice Client</div>
            <div class="subtitle">Direct Full-Duplex Web Audio Streaming (16-bit Linear PCM, Sub-150ms)</div>
        </div>

        <div class="panel">
            <div class="controls-row">
                <div>
                    <label>Persona</label>
                    <select id="persona-select">
                        <option value="muthoot_recovery">Muthoot Fincorp (Loan EMI Recovery)</option>
                        <option value="bank_kyc">Bank of Baroda (Mandatory Re-KYC)</option>
                        <option value="swiggy_delivery">Swiggy Express (Delivery Address Verification)</option>
                    </select>
                </div>
                <div>
                    <label>Language</label>
                    <select id="lang-select">
                        <option value="hi">Hindi (हिंदी)</option>
                        <option value="en">English (India)</option>
                        <option value="gu">Gujarati (ગુજરાતી)</option>
                        <option value="mr">Marathi (मराठी)</option>
                        <option value="ta">Tamil (தமிழ்)</option>
                        <option value="te">Telugu (తెలుగు)</option>
                    </select>
                </div>
            </div>

            <button class="call-btn call-btn-start" id="call-toggle-btn" onclick="toggleCall()">Start Phone Call</button>
            <div style="text-align: center;">
                <span class="status-badge status-idle" id="call-status">Call Disconnected</span>
            </div>

            <canvas id="waveform-canvas"></canvas>

            <div class="meta-chips">
                <span class="chip" id="chip-lang">Language: HI</span>
                <span class="chip" id="chip-sentiment">Sentiment: CALM</span>
                <span class="chip" id="chip-barge">Barge-In: Active</span>
            </div>
        </div>

        <div class="panel">
            <label>Live Call Dialogue & Transcripts</label>
            <div class="transcript-box" id="transcript-box">
                <div class="msg msg-agent">Click 'Start Phone Call' and speak into your microphone. Verbalyze will respond in real time.</div>
            </div>
        </div>
    </div>

    <script>
        let isCallActive = false;
        let ws = null;
        let audioContext = null;
        let micStream = null;
        let scriptProcessor = null;
        let audioQueue = [];
        let isPlayingAudio = false;

        async function toggleCall() {
            if (isCallActive) {
                endCall();
            } else {
                await startCall();
            }
        }

        async function startCall() {
            const persona = document.getElementById('persona-select').value;
            const lang = document.getElementById('lang-select').value;
            const sessionId = 'browser_' + Math.random().toString(36).substr(2, 9);

            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
            } catch(e) {
                alert('Microphone access denied: ' + e);
                return;
            }

            const loc = window.location;
            const wsUrl = (loc.protocol === 'https:' ? 'wss://' : 'ws://') + loc.host + 
                          '/telephony/browser/stream?session_id=' + sessionId + 
                          '&lang=' + lang + '&persona=' + persona + '&sample_rate=16000';

            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                isCallActive = true;
                document.getElementById('call-toggle-btn').textContent = 'End Phone Call';
                document.getElementById('call-toggle-btn').className = 'call-btn call-btn-end';
                document.getElementById('call-status').textContent = 'Connected (Listening)';
                document.getElementById('call-status').className = 'status-badge status-live';
                startMicStreaming();
            };

            ws.onmessage = (evt) => {
                try {
                    const msg = JSON.parse(evt.data);
                    handleServerMessage(msg);
                } catch(e) {}
            };

            ws.onclose = () => {
                endCall();
            };
        }

        function handleServerMessage(msg) {
            const box = document.getElementById('transcript-box');

            if (msg.type === 'transcript') {
                const div = document.createElement('div');
                div.className = msg.role === 'user' ? 'msg msg-user' : 'msg msg-agent';
                div.textContent = (msg.role === 'user' ? 'Caller: ' : 'Agent: ') + msg.text;
                box.appendChild(div);
                box.scrollTop = box.scrollHeight;
            } else if (msg.type === 'turn_complete') {
                if (msg.sentiment) {
                    document.getElementById('chip-sentiment').textContent = 'Sentiment: ' + msg.sentiment.category + ' (' + (msg.sentiment.composite_agitation || 0).toFixed(2) + ')';
                }
                if (msg.language_info) {
                    document.getElementById('chip-lang').textContent = 'Language: ' + (msg.language_info.primary_language || 'hi').toUpperCase();
                }
                if (msg.terminated) {
                    setTimeout(endCall, 2000);
                }
            } else if (msg.type === 'audio') {
                playAudioChunk(msg.payload);
            } else if (msg.type === 'clear') {
                audioQueue = [];
                document.getElementById('call-status').textContent = 'Caller Barge-In Detected';
                document.getElementById('call-status').className = 'status-badge status-live';
            }
        }

        function playAudioChunk(b64) {
            const raw = atob(b64);
            const len = raw.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) bytes[i] = raw.charCodeAt(i);
            const pcm16 = new Int16Array(bytes.buffer);
            const float32 = new Float32Array(pcm16.length);
            for (let i = 0; i < pcm16.length; i++) float32[i] = pcm16[i] / 32768.0;

            audioQueue.push(float32);
            if (!isPlayingAudio) playNextChunk();
        }

        function playNextChunk() {
            if (audioQueue.length === 0 || !audioContext) {
                isPlayingAudio = false;
                return;
            }
            isPlayingAudio = true;
            const chunk = audioQueue.shift();
            const buffer = audioContext.createBuffer(1, chunk.length, 16000);
            buffer.getChannelData(0).set(chunk);
            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            source.onended = () => playNextChunk();
            source.start();
        }

        function startMicStreaming() {
            const source = audioContext.createMediaStreamSource(micStream);
            scriptProcessor = audioContext.createScriptProcessor(512, 1, 1);
            source.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            const canvas = document.getElementById('waveform-canvas');
            const ctx = canvas.getContext('2d');

            scriptProcessor.onaudioprocess = (e) => {
                if (!isCallActive || !ws || ws.readyState !== WebSocket.OPEN) return;
                const input = e.inputBuffer.getChannelData(0);

                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#065f46';
                ctx.beginPath();
                for (let i = 0; i < input.length; i += 4) {
                    const val = input[i] * 30;
                    ctx.lineTo(i, 30 + val);
                }
                ctx.stroke();

                const pcm16 = new Int16Array(input.length);
                for (let i = 0; i < input.length; i++) {
                    const s = Math.max(-1, Math.min(1, input[i]));
                    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                const bytes = new Uint8Array(pcm16.buffer);
                let binary = '';
                for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
                const b64 = btoa(binary);

                ws.send(JSON.stringify({ type: 'audio', payload: b64 }));
            };
        }

        function endCall() {
            isCallActive = false;
            if (ws) { ws.close(); ws = null; }
            if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
            if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
            if (audioContext) { audioContext.close(); audioContext = null; }
            audioQueue = [];
            document.getElementById('call-toggle-btn').textContent = 'Start Phone Call';
            document.getElementById('call-toggle-btn').className = 'call-btn call-btn-start';
            document.getElementById('call-status').textContent = 'Call Disconnected';
            document.getElementById('call-status').className = 'status-badge status-idle';
        }
    </script>
</body>
</html>
"""
