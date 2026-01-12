"use client";

import { useState, useRef } from 'react';
import { FileText, Printer, Check, Clock, Download, Upload, Loader2 } from 'lucide-react';
import Image from 'next/image';

import { jsPDF } from "jspdf";
import { generateGeminiResponse } from "@/app/actions";
import { api, AnalysisResult } from '@/services/api';

const MOCK_ALERTS = [
    { id: 1, type: "Rail Fracture", severity: "High", time: "10:42 AM", location: "KM-42.5", status: "New", score: 98, img: "/mock-images/real_fault_1.jpg" },
    { id: 2, type: "Obstacle", severity: "Medium", time: "09:15 AM", location: "KM-38.2", status: "Investigating", score: 75, img: "/mock-images/real_obstacle_1.jpg" },
    { id: 3, type: "Missing Clip", severity: "Low", time: "Yesterday", location: "KM-40.1", status: "Resolved", score: 45, img: "/mock-images/real_track_1.jpg" },
];

export default function AnalysisPage() {
    const [alerts, setAlerts] = useState<any[]>(MOCK_ALERTS);
    const [selectedAlert, setSelectedAlert] = useState<any>(MOCK_ALERTS[0]);
    const [generating, setGenerating] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files?.length) return;

        setAnalyzing(true);
        const files = Array.from(e.target.files);

        try {
            const result = await api.analyzeCombined(files);
            console.log("Analysis Result:", result);

            // Map API result to Alert format
            let imageUrl = ""; // Empty means show CSV placeholder
            let type = "Unknown Anomaly";
            let score = 0;
            let severity = "Low";
            let isCsvFile = false;
            let isVideo = false; // Track if media is video
            let detections: any[] = []; // For bounding box overlay

            // Extract info from result - check expert_results structure
            if (result.result) {
                const expertResults = result.result.expert_results;
                const visual = expertResults?.visual || result.result.visual_result;
                const structural = expertResults?.structural || result.result.structural_result;
                const tampering = result.result.tampering_analysis?.tampering_assessment;

                // Try to get image URL from visual result - prefer annotated image
                if (visual) {
                    const visItem = Array.isArray(visual) ? visual[0] : visual;

                    // Check if this is a video file
                    if (visItem?.file_type === 'video') {
                        isVideo = true;
                        // For videos, prefer annotated version if available
                        if (visItem.annotated_image_url) {
                            imageUrl = api.getUploadUrl(visItem.annotated_image_url);
                        } else if (visItem.file_url) {
                            imageUrl = api.getUploadUrl(visItem.file_url);
                        }
                    } else {
                        // Use annotated image if available (has bounding boxes drawn)
                        if (visItem && visItem.annotated_image_url) {
                            imageUrl = api.getUploadUrl(visItem.annotated_image_url);
                        } else if (visItem && visItem.file_url) {
                            imageUrl = api.getUploadUrl(visItem.file_url);
                        }
                    }

                    // Extract detections for bounding boxes
                    if (visItem?.detections) {
                        detections = visItem.detections;
                    }

                    // Get type and score from visual tampering info (this is more accurate)
                    if (visItem?.tampering) {
                        const visTampering = visItem.tampering;
                        if (visTampering.tampering_types && visTampering.tampering_types.length > 0) {
                            type = visTampering.tampering_types[0]; // Take first tampering type
                        }
                        if (visTampering.tampering_probability) {
                            score = Math.round(visTampering.tampering_probability * 100);
                        }
                    }

                    // Fallback to risk assessment
                    if (visItem?.risk_assessment) {
                        if (score === 0 && visItem.risk_assessment.confidence) {
                            score = Math.round(visItem.risk_assessment.confidence * 100);
                        }
                        if (visItem.risk_assessment.risk_level) {
                            severity = visItem.risk_assessment.risk_level.charAt(0).toUpperCase() + visItem.risk_assessment.risk_level.slice(1);
                        }
                    }
                }

                // Handle structural/CSV result
                if (structural && !imageUrl) {
                    const strucItem = Array.isArray(structural) ? structural[0] : structural;
                    if (strucItem) {
                        // Mark as CSV and get filename
                        isCsvFile = true;
                        imageUrl = strucItem.file_path || strucItem.file_url || "csv_file";

                        // Extract meaningful info from structural result
                        if (strucItem.result) {
                            const strucResult = strucItem.result;
                            type = strucResult.tampering_detected ? "Vibration Anomaly" : "Normal Condition";
                            score = Math.round((strucResult.confidence || 0) * 100);
                            severity = strucResult.risk_level?.charAt(0).toUpperCase() + strucResult.risk_level?.slice(1) || "Low";
                        }

                        // Use failure ratio for better type description
                        if (strucItem.vibration_analysis?.failure_ratio > 0.5) {
                            type = "Critical Vibration Anomaly";
                        }
                    }
                }

                // Override with top-level tampering info only if we don't have visual info
                if (type === "Unknown Anomaly" && tampering && tampering.tampering_type && tampering.tampering_type !== "Unknown") {
                    type = tampering.tampering_type;
                }
                if (score === 0 && tampering?.confidence) {
                    score = Math.round(tampering.confidence * 100);
                }

                // Get severity from overall assessment
                if (result.result.overall_assessment?.risk_level) {
                    severity = result.result.overall_assessment.risk_level.charAt(0).toUpperCase() + result.result.overall_assessment.risk_level.slice(1);
                } else if (result.result.overall_risk_level) {
                    severity = result.result.overall_risk_level.charAt(0).toUpperCase() + result.result.overall_risk_level.slice(1);
                }
            }

            const newAlert = {
                id: Date.now(), // Generate a temp ID
                type: type,
                severity: severity,
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                location: "Uploaded File",
                status: "New",
                score: score,
                img: imageUrl,
                isVideo: isVideo, // Track if media is video for proper rendering
                detections: detections, // For bounding box overlay
                rawResult: result // Store raw result for detailed view if needed
            };

            setAlerts(prev => [newAlert, ...prev]);
            setSelectedAlert(newAlert);

        } catch (error) {
            console.error("Analysis failed:", error);
            alert("Analysis failed. Please try again.");
        } finally {
            setAnalyzing(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

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
                <div className="flex gap-2">
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileUpload}
                        className="hidden"
                        multiple
                        accept="image/*,video/*,.csv" // Added .csv support
                    />
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={analyzing}
                        className="bg-govt-orange text-white px-4 py-2 rounded flex items-center gap-2 hover:bg-orange-600 disabled:opacity-50"
                    >
                        {analyzing ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                        {analyzing ? "Analyzing..." : "Upload Evidence"}
                    </button>
                    <button className="bg-govt-navy text-white px-4 py-2 rounded flex items-center gap-2 hover:bg-govt-blue">
                        <Download size={16} /> Export Daily Log
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
                {/* Alert List */}
                <div className="bg-white rounded-lg border border-gray-200 overflow-hidden flex flex-col">
                    <div className="p-4 bg-gray-50 border-b border-gray-200 font-bold text-gray-700">
                        Alert Log (Last 24h)
                    </div>
                    <div className="overflow-y-auto flex-1 p-2 space-y-2">
                        {alerts.map(alert => (
                            <div
                                key={alert.id}
                                onClick={() => setSelectedAlert(alert)}
                                className={`p-4 rounded-lg cursor-pointer transition-all border ${selectedAlert.id === alert.id ? 'bg-blue-50 border-govt-blue' : 'bg-white border-transparent hover:bg-gray-50'}`}
                            >
                                <div className="flex justify-between mb-1">
                                    <span className={`text-xs font-bold px-2 py-0.5 rounded ${alert.severity === 'High' || alert.severity === 'Critical' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'}`}>
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
                            <div className="relative rounded-lg overflow-hidden border border-gray-300 group bg-gray-100 flex items-center justify-center min-h-[200px]">
                                {/* Check if it's a CSV file or no image available */}
                                {(!selectedAlert.img || selectedAlert.img.includes('.csv') || selectedAlert.img === 'csv_file') ? (
                                    <div className="p-8 text-center">
                                        <FileText size={48} className="text-blue-500 mx-auto mb-2" />
                                        <p className="text-sm font-medium text-gray-700">Vibration Data Analysis</p>
                                        {selectedAlert.img && selectedAlert.img !== 'csv_file' && (
                                            <p className="font-mono text-xs text-gray-400 mt-1">{selectedAlert.img.split('/').pop()}</p>
                                        )}
                                        <p className="text-xs text-gray-500 mt-2">Structural analysis complete</p>
                                        <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur">
                                            Confidence: {selectedAlert.score}%
                                        </div>
                                    </div>
                                ) : selectedAlert.isVideo ? (
                                    /* Video player for video files */
                                    <div className="relative">
                                        <video
                                            src={selectedAlert.img}
                                            controls
                                            autoPlay
                                            muted
                                            loop
                                            className="w-full h-auto max-h-[400px] object-contain rounded"
                                        >
                                            Your browser does not support the video tag.
                                        </video>
                                        <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur">
                                            Confidence: {selectedAlert.score}%
                                        </div>
                                    </div>
                                ) : (
                                    <div className="relative">
                                        <Image
                                            src={selectedAlert.img}
                                            alt="Evidence"
                                            width={500}
                                            height={300}
                                            className="w-full h-auto object-cover"
                                            unoptimized
                                        />
                                        {/* Bounding boxes are now drawn in the backend-saved annotated image */}
                                        <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur">
                                            Confidence: {selectedAlert.score}%
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-gray-50 p-3 rounded">
                                    <p className="text-xs text-gray-500">Detection Model</p>
                                    <p className="font-mono font-bold text-xs text-gray-700">Multi-Modal Fusion</p>
                                </div>
                                <div className="bg-gray-50 p-3 rounded">
                                    <p className="text-xs text-gray-500">Processing Time</p>
                                    <p className="font-mono font-bold text-xs text-gray-700">~2.5s</p>
                                </div>
                            </div>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <h4 className="font-bold text-gray-800 mb-2">AI Analysis</h4>
                                <p className="text-sm text-gray-600 leading-relaxed">
                                    The system detected a potential <strong>{selectedAlert.type}</strong> at {selectedAlert.location}.
                                    Visual spectrum analysis indicates structural anomaly consistent with {selectedAlert.type.toLowerCase()}.
                                    Immediate inspection recommended.
                                </p>
                            </div>

                            {/* Detection List */}
                            {selectedAlert.detections && selectedAlert.detections.length > 0 && (
                                <div className="bg-gray-50 p-3 rounded border border-gray-200">
                                    <h5 className="font-bold text-xs text-gray-700 mb-2">🔍 Detected Objects ({selectedAlert.detections.length})</h5>
                                    <div className="space-y-1 max-h-[100px] overflow-y-auto">
                                        {selectedAlert.detections.slice(0, 5).map((det: any, idx: number) => (
                                            <div key={idx} className="flex items-center justify-between text-xs">
                                                <span className="flex items-center gap-2">
                                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: det.type === 'person' ? '#ef4444' : det.type === 'foreign_object' ? '#f97316' : '#3b82f6' }} />
                                                    {det.label || det.type}
                                                </span>
                                                <span className="text-gray-500">{Math.round(det.confidence * 100)}%</span>
                                            </div>
                                        ))}
                                        {selectedAlert.detections.length > 5 && <p className="text-xs text-gray-400">+{selectedAlert.detections.length - 5} more...</p>}
                                    </div>
                                </div>
                            )}

                            {/* Recommendations */}
                            {selectedAlert.rawResult?.result?.expert_results?.visual?.recommendations && (
                                <div className="bg-yellow-50 p-3 rounded border border-yellow-200">
                                    <h5 className="font-bold text-xs text-yellow-800 mb-2">⚠️ Actions Required</h5>
                                    <ul className="text-xs text-yellow-700 space-y-1">
                                        {selectedAlert.rawResult.result.expert_results.visual.recommendations.slice(0, 4).map((rec: string, idx: number) => (
                                            <li key={idx}>• {rec}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* Alerts */}
                            {selectedAlert.rawResult?.result?.expert_results?.visual?.alerts && (
                                <div className="bg-red-50 p-3 rounded border border-red-200">
                                    <h5 className="font-bold text-xs text-red-800 mb-2">🚨 System Alerts</h5>
                                    <ul className="text-xs text-red-700 space-y-1">
                                        {selectedAlert.rawResult.result.expert_results.visual.alerts.map((alert: string, idx: number) => (
                                            <li key={idx}>{alert}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* Risk Level Badge */}
                            <div className={`p-3 rounded border ${selectedAlert.severity === 'Critical' ? 'bg-red-100 border-red-300' : selectedAlert.severity === 'High' ? 'bg-orange-100 border-orange-300' : 'bg-gray-100 border-gray-300'}`}>
                                <div className="flex justify-between items-center">
                                    <span className="text-xs font-medium text-gray-700">Risk Level</span>
                                    <span className={`text-xs font-bold px-2 py-1 rounded ${selectedAlert.severity === 'Critical' ? 'bg-red-500 text-white' : selectedAlert.severity === 'High' ? 'bg-orange-500 text-white' : 'bg-gray-500 text-white'}`}>
                                        {selectedAlert.severity}
                                    </span>
                                </div>
                            </div>

                            <div className="bg-gray-50 p-3 rounded border border-gray-200 flex items-center gap-3">
                                <div className="w-4 h-4 rounded-full border-2 border-dashed border-gray-400"></div>
                                <div>
                                    <p className="text-xs font-bold text-gray-500">Officer Review</p>
                                    <p className="text-[10px] text-gray-400">{selectedAlert.status}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
