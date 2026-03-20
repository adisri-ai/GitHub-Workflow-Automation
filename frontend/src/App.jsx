import React, { useState, useRef, useEffect } from "react";
import { Terminal, Send, Command, Trash2, Loader2, Code, ChevronRight, Mic, MicOff, Cpu, Lock, ExternalLink, RotateCcw } from "lucide-react";

const GitPilotLogo = () => (
  <div className="relative group">
    <div className="absolute -inset-1 bg-gradient-to-r from-[#00ff9d] to-cyan-500 rounded-xl blur opacity-25 group-hover:opacity-75 transition duration-1000"></div>
    <img src="GitPilotLogo.png" className="relative w-16 h-16 object-contain rounded-xl border border-slate-700 bg-slate-900/50 shadow-2xl" alt="GitPilot Logo" />
  </div>
);

function App() {
  const [prompt, setPrompt] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isListening, setIsListening] = useState(false);
  
  // Auth & Undo States
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);
  const [authCode, setAuthCode] = useState(null);
  const [canUndo, setCanUndo] = useState(false); // NEW STATE

  const chatEndRef = useRef(null);

  useEffect(() => { checkAuth(); }, []);

  const checkAuth = async () => {
    try {
      const res = await fetch("http://localhost:5000/api/auth/status");
      const data = await res.json();
      setIsAuthenticated(data.authenticated);
      setCanUndo(data.undo_available); // Update undo state on load
    } catch (e) { console.error("Auth check failed", e); } 
    finally { setIsCheckingAuth(false); }
  };

  const startLogin = async () => {
    try {
      const res = await fetch("http://localhost:5000/api/auth/login", { method: "POST" });
      const data = await res.json();
      if (data.user_code) setAuthCode(data);
    } catch (e) { alert("Failed to start login flow"); }
  };

  const finishLogin = async () => {
    await checkAuth();
    if (!isAuthenticated) alert("Not logged in yet.");
  };

  // --- UNDO HANDLER ---
  const handleUndo = async () => {
    if (!canUndo || isProcessing) return;
    
    const userMsg = { type: "user", content: "↺ Undo Last Action", timestamp: new Date().toLocaleTimeString() };
    const agentMsg = { type: "agent", logs: [], status: "thinking", timestamp: new Date().toLocaleTimeString() };
    setHistory((prev) => [...prev, userMsg, agentMsg]);
    setIsProcessing(true);

    try {
        const res = await fetch("http://localhost:5000/api/undo", { method: "POST" });
        const data = await res.json();
        
        setHistory((prev) => {
            const newHistory = [...prev];
            const lastMsg = newHistory[newHistory.length - 1];
            if (data.status === "success") {
                lastMsg.logs.push({ type: 'info', text: `Undoing action: ${data.action}` });
                lastMsg.logs.push({ type: 'json_result', data: data.output, label: "Undo Output" });
                lastMsg.status = "completed";
                setCanUndo(data.remaining_undo > 0);
            } else {
                lastMsg.logs.push({ type: 'error', text: data.error || "Undo failed" });
                lastMsg.status = "error";
            }
            return newHistory;
        });
    } catch (e) {
        console.error(e);
    } finally {
        setIsProcessing(false);
    }
  };

  // --- HISTORY & CHAT ---
  const [history, setHistory] = useState(() => {
    const saved = sessionStorage.getItem("gitpilot_history");
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => { sessionStorage.setItem("gitpilot_history", JSON.stringify(history)); scrollToBottom(); }, [history]);
  const scrollToBottom = () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  const clearHistory = () => { setHistory([]); sessionStorage.removeItem("gitpilot_history"); };

  // --- WEB SPEECH ---
  const toggleListening = () => {
    if (isListening) { setIsListening(false); return; }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) { alert("Use Chrome/Edge/Safari"); return; }
    const recognition = new SpeechRecognition();
    recognition.continuous = false; recognition.lang = 'en-US'; recognition.interimResults = false;
    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onresult = (e) => setPrompt(e.results[0][0].transcript);
    recognition.start();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isProcessing) return;

    const userMsg = { type: "user", content: prompt, timestamp: new Date().toLocaleTimeString() };
    const agentMsg = { type: "agent", logs: [], status: "thinking", timestamp: new Date().toLocaleTimeString() };
    setHistory((prev) => [...prev, userMsg, agentMsg]);
    setPrompt("");
    setIsProcessing(true);

    try {
      const response = await fetch("http://localhost:5000/api/agent", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: userMsg.content }),
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split("\n\n");
        lines.forEach((line) => {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.replace("data: ", ""));
              setHistory((prev) => {
                const newHistory = [...prev];
                const lastMsg = newHistory[newHistory.length - 1];
                if (lastMsg.type === "agent") {
                  if (data.type === 'plan') lastMsg.logs.push({ type: 'json_plan', data: data.tasks, label: "Execution Plan" });
                  else if (data.type === 'result') lastMsg.logs.push({ type: 'json_result', data: data.output, label: "Output" });
                  else if (data.type === 'error') { lastMsg.logs.push({ type: 'error', text: data.message }); lastMsg.status = "error"; }
                  else if (data.type === 'done') {
                      lastMsg.status = "completed";
                      if (data.undo_available !== undefined) setCanUndo(data.undo_available);
                  }
                  else lastMsg.logs.push({ type: 'info', text: data.message });
                }
                return newHistory;
              });
            } catch (err) {}
          }
        });
      }
    } catch (error) {
      setHistory(prev => {
        const newHistory = [...prev];
        newHistory[newHistory.length -1].logs.push({ type: 'error', text: "Connection Failed" });
        newHistory[newHistory.length -1].status = "error";
        return newHistory;
      });
    } finally { setIsProcessing(false); }
  };

  // --- RENDER ---
  if (!isCheckingAuth && !isAuthenticated) {
    return (
      <div className="min-h-screen bg-[#0B1120] flex items-center justify-center text-slate-200 font-sans p-4 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 to-[#020617]">
        <div className="bg-[#1e293b]/80 backdrop-blur-md p-8 rounded-2xl border border-slate-700 shadow-2xl max-w-md w-full text-center">
          <div className="flex justify-center mb-4"><GitPilotLogo /></div>
          <h2 className="text-2xl font-bold mt-4 text-white">Authentication Required</h2>
          {!authCode ? (
            <button onClick={startLogin} className="mt-8 w-full py-3 bg-[#00ff9d] text-slate-900 font-bold rounded-lg hover:bg-[#00e08b] transition flex justify-center gap-2 items-center"><Lock size={18} /> Authenticate with GitHub</button>
          ) : (
            <div className="mt-6 bg-[#0f172a] p-5 rounded-lg border border-slate-600 animate-in fade-in">
              <p className="text-xs text-slate-400 mb-2 uppercase">1. Copy Code</p>
              <div className="text-3xl font-mono font-bold text-[#00ff9d] tracking-widest mb-4 select-all">{authCode.user_code}</div>
              <p className="text-xs text-slate-400 mb-2 uppercase">2. Paste at GitHub</p>
              <a href={authCode.verification_uri} target="_blank" rel="noreferrer" className="text-cyan-400 underline flex items-center justify-center gap-1 mb-6">{authCode.verification_uri} <ExternalLink size={14}/></a>
              <button onClick={finishLogin} className="w-full py-3 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-500">I have entered the code</button>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0B1120] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-[#0B1120] to-[#020617] text-slate-200 font-sans selection:bg-[#00ff9d] selection:text-black flex flex-col items-center relative overflow-hidden">
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none"></div>

      <header className="w-full max-w-6xl p-6 flex items-center justify-between border-b border-slate-800/60 bg-[#0B1120]/70 backdrop-blur-md sticky top-0 z-20 shadow-lg">
        <div className="flex items-center gap-5"><GitPilotLogo /><div className="flex flex-col"><h1 className="text-4xl font-extrabold tracking-tighter text-white drop-shadow-lg">Git<span className="text-transparent bg-clip-text bg-gradient-to-r from-[#00ff9d] to-cyan-400">Pilot</span></h1><div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-[#00ff9d] animate-pulse"></span><span className="text-[11px] font-mono text-cyan-500 uppercase tracking-[0.2em]">System Online</span></div></div></div>
        <div className="flex items-center gap-4"><button onClick={clearHistory} className="px-4 py-2 rounded-lg text-xs font-medium flex items-center gap-2 text-slate-400 hover:text-red-400 border border-transparent hover:border-red-500/20"><Trash2 size={14} /> RESET</button></div>
      </header>

      <main className="flex-1 w-full max-w-5xl p-4 md:p-8 overflow-y-auto space-y-10 pb-40 z-10 custom-scrollbar">
        {history.length === 0 && (
          <div className="h-[60vh] flex flex-col items-center justify-center text-slate-600 animate-in fade-in zoom-in duration-500"><div className="p-8 rounded-full bg-slate-900/50 border border-slate-800 mb-6 shadow-2xl"><Terminal size={80} className="text-slate-700" /></div><p className="text-2xl font-bold text-slate-400">System Initialized.</p></div>
        )}
        {history.map((msg, idx) => (
          <div key={idx} className={`flex flex-col ${msg.type === 'user' ? 'items-end' : 'items-start'} animate-in slide-in-from-bottom-5 duration-300`}>
            <span className="text-[10px] font-bold text-slate-500 mb-2 px-1 uppercase">{msg.type === 'user' ? 'You' : 'GitPilot'} • {msg.timestamp}</span>
            {msg.type === 'user' ? (
              <div className="bg-gradient-to-br from-blue-600 to-blue-700 text-white px-6 py-4 rounded-2xl rounded-tr-sm max-w-[85%] border border-blue-500/30 shadow-lg"><p className="whitespace-pre-wrap font-medium leading-relaxed">{msg.content}</p></div>
            ) : (
              <div className="w-full max-w-[95%] bg-[#0f1219]/95 border border-slate-700/50 rounded-xl overflow-hidden shadow-2xl relative backdrop-blur-sm group">
                <div className="bg-[#1e293b]/50 px-4 py-3 flex items-center justify-between border-b border-slate-700/50"><div className="flex gap-2"><div className="w-3 h-3 rounded-full bg-red-500/80"></div><div className="w-3 h-3 rounded-full bg-yellow-500/80"></div><div className="w-3 h-3 rounded-full bg-green-500/80"></div></div><div className="text-xs font-mono text-cyan-400/80 flex items-center gap-2">{msg.status === 'thinking' ? <><Loader2 size={12} className="animate-spin" /> EXECUTING</> : <><Code size={12} /> LOG</>}</div></div>
                <div className="p-6 font-mono text-sm space-y-4">
                  {msg.logs.map((log, lIdx) => (
                    <div key={lIdx} className="break-words">
                      {log.type === 'info' && <div className="flex gap-3 text-slate-400 items-start"><ChevronRight size={16} className="mt-0.5 text-cyan-600 shrink-0"/><span>{log.text}</span></div>}
                      {log.type === 'json_plan' && <div className="my-4"><div className="text-[#00ff9d] text-xs font-bold mb-2 flex items-center gap-2 bg-[#00ff9d]/5 w-fit px-2 py-1 rounded"><Code size={12} /> {log.label}</div><pre className="bg-[#050505] p-4 rounded-lg border-l-2 border-[#00ff9d] text-xs text-slate-300 overflow-x-auto custom-scrollbar">{JSON.stringify(log.data, null, 2)}</pre></div>}
                      {log.type === 'json_result' && <div className="my-4"><div className="text-cyan-400 text-xs font-bold mb-2 flex items-center gap-2 bg-cyan-500/5 w-fit px-2 py-1 rounded"><Terminal size={12} /> {log.label}</div><pre className="bg-[#050505] p-4 rounded-lg border-l-2 border-cyan-500 text-xs text-slate-300 overflow-x-auto custom-scrollbar">{JSON.stringify(log.data, null, 2)}</pre></div>}
                      {log.type === 'error' && <div className="flex gap-3 text-red-400 bg-red-950/20 p-3 rounded-lg border border-red-900/50 mt-2"><span className="font-bold">✖</span> <span>{log.text}</span></div>}
                    </div>
                  ))}
                  {msg.status === 'completed' && <div className="mt-6 pt-4 border-t border-slate-800 text-[#00ff9d] text-xs flex items-center gap-2 font-bold tracking-wide"><div className="w-2 h-2 bg-[#00ff9d] rounded-full shadow-[0_0_10px_#00ff9d]"></div>SEQUENCE COMPLETED</div>}
                </div>
              </div>
            )}
          </div>
        ))}
        <div ref={chatEndRef} />
      </main>

      <footer className="fixed bottom-0 w-full bg-[#0B1120]/80 backdrop-blur-xl border-t border-slate-800 p-6 flex justify-center z-30 shadow-[0_-10px_40px_rgba(0,0,0,0.5)]">
        <form onSubmit={handleSubmit} className="w-full max-w-4xl relative group">
          <div className={`absolute -inset-0.5 bg-gradient-to-r from-[#00ff9d] via-cyan-500 to-purple-600 rounded-xl opacity-30 blur transition duration-1000 group-hover:opacity-60 ${isListening ? 'opacity-100 animate-pulse' : ''}`}></div>
          <div className="relative flex items-center bg-[#0f172a] rounded-xl border border-slate-700 shadow-2xl">
            <Command className="w-5 h-5 ml-4 text-slate-500" />
            <input type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={isListening ? "Listening..." : "Create a private repo 'nexus-core'..."} className="w-full bg-transparent border-none text-slate-100 px-4 py-4 focus:ring-0 font-medium tracking-wide placeholder-slate-600" disabled={isProcessing} />
            
            {/* UNDO BUTTON */}
            <button 
              type="button" 
              onClick={handleUndo} 
              disabled={!canUndo || isProcessing}
              title="Undo last action"
              className={`p-3 rounded-lg transition-all duration-300 ${canUndo ? 'text-yellow-500 hover:bg-yellow-500/10' : 'text-slate-700 cursor-not-allowed'}`}
            >
              <RotateCcw className="w-5 h-5" />
            </button>

            <button type="button" onClick={toggleListening} className={`p-3 mr-2 rounded-lg transition-all duration-300 ${isListening ? 'bg-red-500/20 text-red-500 shadow-[0_0_15px_rgba(239,68,68,0.4)]' : 'hover:bg-slate-800 text-slate-400 hover:text-white'}`}><Mic className="w-5 h-5" /></button>
            <button type="submit" disabled={!prompt || isProcessing} className={`mr-2 p-3 rounded-lg transition-all duration-300 ${prompt && !isProcessing ? 'bg-[#00ff9d] text-slate-900 hover:bg-[#00e08b] shadow-[0_0_20px_rgba(0,255,157,0.4)] font-bold' : 'bg-slate-800 text-slate-600 cursor-not-allowed'}`}>{isProcessing ? <Loader2 className="w-5 h-5 animate-spin"/> : <Send className="w-5 h-5" />}</button>
          </div>
        </form>
      </footer>
    </div>
  );
}

export default App;