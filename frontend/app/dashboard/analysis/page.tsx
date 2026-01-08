"use client";

import { useState } from 'react';
import { FileText, Printer, Check, Clock, Download } from 'lucide-react';
import Image from 'next/image';

import { jsPDF } from "jspdf";
import { generateGeminiResponse } from "@/app/actions";

const MOCK_ALERTS = [
    { id: 1, type: "Rail Fracture", severity: "High", time: "10:42 AM", location: "KM-42.5", status: "New", score: 98, img: "/mock-images/real_fault_1.jpg" },
    { id: 2, type: "Obstacle", severity: "Medium", time: "09:15 AM", location: "KM-38.2", status: "Investigating", score: 75, img: "/mock-images/real_obstacle_1.jpg" },
    { id: 3, type: "Missing Clip", severity: "Low", time: "Yesterday", location: "KM-40.1", status: "Resolved", score: 45, img: "/mock-images/real_track_1.jpg" },
];

export default function AnalysisPage() {
    const [selectedAlert, setSelectedAlert] = useState(MOCK_ALERTS[0]);
    const [generating, setGenerating] = useState(false);

    const handleGenerateReport = async () => {
        setGenerating(true);

        const prompt = `
            Generate a DETAILED, formal Indian Railways Technical Incident Report for:
            - TYPE: ${selectedAlert.type}
            - SEVERITY: ${selectedAlert.severity}
            - LOCATION: ${selectedAlert.location}
            
            Please provide a comprehensive analysis with the following structured sections:
            1. EXECUTIVE SUMMARY: A high-level overview of the detection.
            2. TECHNICAL FAULT CHARACTERISTICS: Observations from the visual spectrum analysis (fracture patterns, obstacle dimensions, etc.).
            3. RISK ASSESSMENT: Potential impact on train operations, track stability, and passenger safety.
            4. REMEDIAL ACTION PLAN: Immediate engineering steps (e.g., speed restrictions, weld repair, obstacle removal).
            5. PREVENTATIVE RECOMMENDATIONS: Future measures to avoid recurrence.
            
            Use professional railway engineering terminology. Surround headers with **.
            Total word count: 400-500 words (to fit perfectly on 2 pages).
        `;

        const loadImage = (url: string): Promise<string> => {
            return new Promise((resolve) => {
                const img = new window.Image();
                img.crossOrigin = 'anonymous';
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx?.drawImage(img, 0, 0);
                    resolve(canvas.toDataURL('image/png'));
                };
                img.onerror = () => resolve('');
                img.src = url;
            });
        };

        try {
            const reportText = await generateGeminiResponse(prompt);
            const doc = new jsPDF();
            const pageWidth = doc.internal.pageSize.getWidth();
            const pageHeight = doc.internal.pageSize.getHeight();
            const margin = 20;
            const contentWidth = pageWidth - (margin * 2);
            let yPos = 15;

            // 1. Pre-load Assets
            const irLogoData = await loadImage('/mock-images/Indian_Railways_Vector_Logo.png');
            const goiLogoData = await loadImage('/mock-images/Government_of_India_logo.svg');
            const incidentImageData = await loadImage(selectedAlert.img);

            // 2. High-End Official Header
            if (goiLogoData) {
                doc.addImage(goiLogoData, 'PNG', (pageWidth / 2) - 12, yPos, 24, 18);
                yPos += 22;
            }

            doc.setFontSize(10);
            doc.setFont("helvetica", "bold");
            doc.setTextColor(60, 60, 60);
            doc.text("GOVERNMENT OF INDIA", pageWidth / 2, yPos, { align: "center" });
            yPos += 6;
            doc.setFontSize(14);
            doc.setTextColor(0, 31, 63);
            doc.text("MINISTRY OF RAILWAYS", pageWidth / 2, yPos, { align: "center" });

            yPos += 4;
            doc.setDrawColor(0, 31, 63);
            doc.setLineWidth(0.8);
            doc.line(margin, yPos, pageWidth - margin, yPos);
            yPos += 8;

            // 3. Metadata Header (Horizontal Bar)
            doc.setFillColor(0, 31, 63);
            doc.rect(margin, yPos, contentWidth, 12, 'F');
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(9);
            doc.setFont("helvetica", "bold");
            doc.text(`INCIDENT REPORT: #INC-2026-${selectedAlert.id}`, margin + 5, yPos + 8);
            doc.text(`SEVERITY: ${selectedAlert.severity.toUpperCase()}`, pageWidth / 2, yPos + 8, { align: 'center' });
            doc.text(`DATE: ${new Date().toLocaleDateString()}`, pageWidth - margin - 5, yPos + 8, { align: 'right' });

            yPos += 22;

            // 4. Primary Evidence Image (Large & Professional)
            if (incidentImageData) {
                doc.setTextColor(0);
                doc.setFontSize(11);
                doc.text("EXHIBIT I: PRIMARY SITE VISUAL", margin, yPos);
                yPos += 5;
                doc.addImage(incidentImageData, 'JPEG', margin, yPos, contentWidth, 70);

                // Caption
                doc.setFillColor(245, 245, 245);
                doc.rect(margin, yPos + 70, contentWidth, 8, 'F');
                doc.setFontSize(8);
                doc.setFont("helvetica", "italic");
                doc.setTextColor(100);
                doc.text(`Location: ${selectedAlert.location} | Confidence Score: ${selectedAlert.score}% | Sensor: RSOD-Drone-V3`, margin + 5, yPos + 75);

                yPos += 90;
            }

            // 5. Technical Analysis Sections
            doc.setTextColor(0);
            doc.setFontSize(12);
            doc.setFont("helvetica", "bold");
            doc.text("TECHNICAL EVALUATION & REMEDIAL ACTIONS", margin, yPos);
            yPos += 10;

            doc.setFont("helvetica", "normal");
            doc.setFontSize(10.5);
            doc.setTextColor(30);

            const lines = doc.splitTextToSize(reportText?.replace(/\*\*/g, '') || "Technical data extraction failed.", contentWidth);

            lines.forEach((line: string) => {
                if (yPos > pageHeight - 30) {
                    // Page limit check
                    if (doc.internal.pages.length >= 3) return; // jsPDF pages are 1-indexed for first page, plus current

                    doc.addPage();
                    yPos = 25;
                    // Mini header for second page
                    doc.setFontSize(8);
                    doc.setTextColor(150);
                    doc.text(`Safety Report #INC-2026-${selectedAlert.id}`, margin, 15);
                    doc.text("Page 2 of 2", pageWidth - margin, 15, { align: 'right' });

                    doc.setTextColor(30);
                    doc.setFontSize(10.5);
                    doc.setFont("helvetica", "normal");
                }
                doc.text(line, margin, yPos);
                yPos += 7;
            });

            // 6. Signature & Footer (Always on bottom of whatever page it is)
            if (yPos > pageHeight - 45) {
                // If not enough space, just push to bottom of current or last page
                yPos = pageHeight - 45;
            } else {
                yPos = pageHeight - 45;
            }

            doc.setDrawColor(200);
            doc.setLineWidth(0.2);
            doc.line(margin, yPos, pageWidth - margin, yPos);

            yPos += 10;
            // IR Logo at bottom
            if (irLogoData) {
                doc.addImage(irLogoData, 'PNG', margin, yPos, 15, 15);
            }

            doc.setFontSize(8);
            doc.setFont("helvetica", "bold");
            doc.setTextColor(0);
            doc.text("AUTHORIZED BY:", margin + 20, yPos + 5);
            doc.setFont("helvetica", "normal");
            doc.text("CHIEF SAFETY OFFICER (NORTHERN RAILWAYS)", margin + 20, yPos + 9);
            doc.text("E-Verified via RSOD Protocol", margin + 20, yPos + 13);

            doc.save(`IR_Detailed_Report_INC2026_${selectedAlert.id}.pdf`);

        } catch (e) {
            console.error(e);
            alert("Error generating detailed report.");
        }
        setGenerating(false);
    };

    return (
        <div className="h-full flex flex-col space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-2xl font-bold text-govt-navy">Analysis & Reports</h2>
                    <p className="text-gray-500">Post-incident Review and Investigation</p>
                </div>
                <button className="bg-govt-navy text-white px-4 py-2 rounded flex items-center gap-2 hover:bg-govt-blue">
                    <Download size={16} /> Export Daily Log
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
                {/* Alert List */}
                <div className="bg-white rounded-lg border border-gray-200 overflow-hidden flex flex-col">
                    <div className="p-4 bg-gray-50 border-b border-gray-200 font-bold text-gray-700">
                        Alert Log (Last 24h)
                    </div>
                    <div className="overflow-y-auto flex-1 p-2 space-y-2">
                        {MOCK_ALERTS.map(alert => (
                            <div
                                key={alert.id}
                                onClick={() => setSelectedAlert(alert)}
                                className={`p-4 rounded-lg cursor-pointer transition-all border ${selectedAlert.id === alert.id ? 'bg-blue-50 border-govt-blue' : 'bg-white border-transparent hover:bg-gray-50'}`}
                            >
                                <div className="flex justify-between mb-1">
                                    <span className={`text-xs font-bold px-2 py-0.5 rounded ${alert.severity === 'High' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'}`}>
                                        {alert.severity}
                                    </span>
                                    <span className="text-xs text-gray-400">{alert.time}</span>
                                </div>
                                <h4 className="font-bold text-gray-800">{alert.type}</h4>
                                <p className="text-xs text-gray-500">{alert.location}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Detail View */}
                <div className="lg:col-span-2 bg-white rounded-lg border border-gray-200 flex flex-col">
                    <div className="p-6 border-b border-gray-100 flex justify-between items-center">
                        <h3 className="text-xl font-bold text-govt-navy">Incident Details: #INC-{selectedAlert.id}</h3>
                        <div className="flex gap-2">
                            <button className="border border-gray-300 text-gray-700 px-3 py-1 rounded text-sm hover:bg-gray-50 flex items-center gap-2">
                                <Printer size={14} /> Print
                            </button>
                            <button
                                onClick={handleGenerateReport}
                                disabled={generating}
                                className={`bg-govt-green text-white px-3 py-1 rounded text-sm hover:bg-green-700 flex items-center gap-2 ${generating ? 'opacity-50' : ''}`}
                            >
                                <FileText size={14} /> {generating ? 'Generating...' : 'Generate Report'}
                            </button>
                        </div>
                    </div>

                    {/* Detail Content */}
                    <div className="p-6 grid grid-cols-2 gap-8">
                        <div className="space-y-4">
                            <div className="relative rounded-lg overflow-hidden border border-gray-300 group">
                                <Image
                                    src={selectedAlert.img}
                                    alt="Evidence"
                                    width={500}
                                    height={300}
                                    className="w-full h-auto object-cover"
                                />
                                <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur">
                                    Confidence: {selectedAlert.score}%
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-gray-50 p-3 rounded">
                                    <p className="text-xs text-gray-500">Detection Model</p>
                                    <p className="font-mono font-bold text-xs text-gray-700">YOLOv8-Rail-S</p>
                                </div>
                                <div className="bg-gray-50 p-3 rounded">
                                    <p className="text-xs text-gray-500">Processing Time</p>
                                    <p className="font-mono font-bold text-xs text-gray-700">42ms</p>
                                </div>
                            </div>
                        </div>

                        <div className="space-y-6">
                            <div>
                                <h4 className="font-bold text-gray-800 mb-2">AI Analysis</h4>
                                <p className="text-sm text-gray-600 leading-relaxed">
                                    The system detected a potential <strong>{selectedAlert.type}</strong> at {selectedAlert.location}.
                                    Visual spectrum analysis indicates structural anomaly consistent with {selectedAlert.id === 1 ? 'metal fatigue/fracture' : 'foreign object obstruction'}.
                                    Immediate inspection recommended.
                                </p>
                            </div>

                            <div className="bg-gray-50 p-3 rounded border border-gray-200 flex items-center gap-3">
                                <div className="w-4 h-4 rounded-full border-2 border-dashed border-gray-400"></div>
                                <div>
                                    <p className="text-xs font-bold text-gray-500">Officer Review</p>
                                    <p className="text-[10px] text-gray-400">Pending</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
