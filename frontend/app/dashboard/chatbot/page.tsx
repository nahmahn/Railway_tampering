"use client";

import { useState } from 'react';
import { Send, User, Bot, Briefcase } from 'lucide-react';
import { generateGeminiResponse } from '@/app/actions';
import ReactMarkdown from 'react-markdown';

const SUGGESTED_QUERIES = [
    "Why was alert #102 flagged?",
    "Show similar incidents in Sector 4",
    "Generate inspection checklist",
];

export default function ChatbotPage() {
    const [messages, setMessages] = useState([
        { role: 'bot', content: "I am the AI Operational Assistant for Indian Railways. I have access to real-time track telemetry and historical data. How can I assist you with the latest alerts?" }
    ]);
    const [input, setInput] = useState('');
    const [activeTab, setActiveTab] = useState('chat');
    const [loading, setLoading] = useState(false);

    const sendMessage = async (text: string) => {
        if (!text.trim()) return;

        const userMsg = { role: 'user', content: text };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        // Context Prompt Engineering
        const systemContext = `
            You are an expert AI assistant for Indian Railways Tampering Detection System.
            You are talking to a railway official.
            Current Context:
- Active Alert: INC - 2026-001
    - Type: Rail Fracture(Suspected)
        - Confidence: 92 %
            - Location: Sector 4, KM - 30
                - Sensor Data: Vibration Spike 42Hz
            
            Answer the user's question professionally, concisely, and using technical railway terminology where appropriate.
            If asked for a checklist, provide a numbered list.
        `;

        try {
            const aiResponseText = await generateGeminiResponse(`${systemContext} \n\nUser: ${text} `);
            setMessages(prev => [...prev, { role: 'bot', content: aiResponseText }]);
        } catch (e) {
            setMessages(prev => [...prev, { role: 'bot', content: "Error connecting to AI service." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="h-full flex flex-col space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-2xl font-bold text-govt-navy">AI Assistant</h2>
                    <p className="text-gray-500">Operational Decision Support</p>
                </div>

                {/* Toggle Mode */}
                <div className="bg-white border md:flex hidden border-gray-300 rounded-lg p-1">
                    <button
                        onClick={() => setActiveTab('chat')}
                        className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'chat' ? 'bg-govt-navy text-white shadow' : 'text-gray-600 hover:bg-gray-100'}`}
                    >
                        Conversation Mode
                    </button>
                    <button
                        onClick={() => setActiveTab('planning')}
                        className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'planning' ? 'bg-govt-navy text-white shadow' : 'text-gray-600 hover:bg-gray-100'}`}
                    >
                        Planning & Logistics
                    </button>
                </div>
            </div>

            <div className="flex-1 bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden flex">
                {activeTab === 'chat' && (
                    <>
                        {/* Chat Area */}
                        <div className="flex-1 flex flex-col">
                            <div className="flex-1 p-6 overflow-y-auto space-y-6">
                                {messages.map((msg, idx) => (
                                    <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                                        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === 'bot' ? 'bg-govt-blue text-white' : 'bg-gray-200 text-gray-700'}`}>
                                            {msg.role === 'bot' ? <Bot size={18} /> : <User size={18} />}
                                        </div>
                                        <div className={`max-w-[70%] p-4 rounded-lg text-sm leading-relaxed ${msg.role === 'bot' ? 'bg-gray-100 text-gray-800' : 'bg-govt-blue text-white'}`}>
                                            {msg.role === 'bot' ? (
                                                <div className="prose prose-sm max-w-none prose-p:leading-relaxed prose-li:my-0 shadow-sm">
                                                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                                                </div>
                                            ) : (
                                                <div className="whitespace-pre-wrap">{msg.content}</div>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>

                            <div className="p-4 border-t border-gray-200 bg-gray-50">
                                <div className="flex gap-2 mb-4 overflow-x-auto pb-2">
                                    {SUGGESTED_QUERIES.map(q => (
                                        <button
                                            key={q}
                                            onClick={() => sendMessage(q)}
                                            className="px-3 py-1 bg-white border border-govt-blue text-govt-blue text-xs rounded-full hover:bg-blue-50 whitespace-nowrap"
                                        >
                                            {q}
                                        </button>
                                    ))}
                                </div>
                                <div className="flex gap-2">
                                    <input
                                        value={input}
                                        onChange={(e) => setInput(e.target.value)}
                                        onKeyDown={(e) => e.key === 'Enter' && sendMessage(input)}
                                        className="flex-1 border border-gray-300 rounded px-4 py-2 text-sm focus:outline-none focus:border-govt-blue"
                                        placeholder="Ask about anomalies, regulations, or historical data..."
                                        disabled={loading}
                                    />
                                    <button
                                        onClick={() => sendMessage(input)}
                                        disabled={loading}
                                        className={`bg-govt-navy text-white px-4 py-2 rounded hover:bg-govt-blue ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                                    >
                                        <Send size={18} />
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Left Context (Desktop) */}
                        <div className="w-80 border-l border-gray-200 bg-gray-50 p-6 hidden lg:block">
                            <h4 className="font-bold text-gray-700 mb-4">Current Context</h4>
                            <div className="bg-white p-4 rounded border border-gray-200 shadow-sm mb-4">
                                <p className="text-xs text-gray-500 uppercase">Active Alert</p>
                                <p className="font-bold text-govt-navy">#INC-2026-001</p>
                            </div>

                            <h4 className="font-bold text-gray-700 mb-4 mt-8">Sensor Data</h4>
                            <ul className="space-y-2 text-sm text-gray-600">
                                <li className="flex justify-between"><span>Vibration:</span> <span className="font-mono font-bold">42Hz (High)</span></li>
                                <li className="flex justify-between"><span>Visual:</span> <span className="font-mono font-bold">Confirmed</span></li>
                                <li className="flex justify-between"><span>GPS:</span> <span className="font-mono">28.6°N 77.2°E</span></li>
                            </ul>
                        </div>
                    </>
                )}

                {activeTab === 'planning' && (
                    <div className="p-8 w-full">
                        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                            <Briefcase className="text-govt-orange" /> Inspection Mission Plan
                        </h3>

                        <div className="grid grid-cols-2 gap-8">
                            <div className="space-y-4">
                                <h4 className="font-bold border-b pb-2">Crew Requirements</h4>
                                <ul className="list-disc pl-5 space-y-2 text-sm text-gray-700">
                                    <li>Senior Section Engineer (P.Way)</li>
                                    <li>2 Trackmen with safety gear</li>
                                    <li>Signal Technician</li>
                                </ul>
                            </div>
                            <div className="space-y-4">
                                <h4 className="font-bold border-b pb-2">Equipment Checklist</h4>
                                <ul className="list-disc pl-5 space-y-2 text-sm text-gray-700">
                                    <li>Ultrasonic Flaw Detector (USFD)</li>
                                    <li>Hydraulic Jack</li>
                                    <li>Fish plates and bolts (Spare)</li>
                                    <li>Red Flag and Flares</li>
                                </ul>
                            </div>
                        </div>

                        <div className="mt-8">
                            <button className="bg-govt-green text-white px-6 py-3 rounded font-bold shadow hover:bg-green-700">
                                Dispatch Maintenance Team
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
