"use client";

import { useState, useRef, useEffect } from 'react';
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

    // Load persisted state on mount
    useEffect(() => {
        const savedAlerts = localStorage.getItem('analysis_alerts');
        const savedSelectedAlert = localStorage.getItem('analysis_selectedAlert');

        if (savedAlerts) {
            try {
                const parsedAlerts = JSON.parse(savedAlerts);
                // Only set if valid array
                if (Array.isArray(parsedAlerts) && parsedAlerts.length > 0) {
                    setAlerts(parsedAlerts);
                }
            } catch (e) { console.error("Failed to parse saved alerts", e); }
        }

        if (savedSelectedAlert) {
            try {
                const parsedSelected = JSON.parse(savedSelectedAlert);
                if (parsedSelected) setSelectedAlert(parsedSelected);
            } catch (e) { console.error("Failed to parse saved selection", e); }
        }
    }, []);

    // Save state whenever it changes (debouncing could be good but direct is fine for now)
    useEffect(() => {
        if (alerts.length > 0 && alerts !== MOCK_ALERTS) {
            localStorage.setItem('analysis_alerts', JSON.stringify(alerts));
        }
    }, [alerts]);

    useEffect(() => {
        if (selectedAlert) {
            localStorage.setItem('analysis_selectedAlert', JSON.stringify(selectedAlert));
        }
    }, [selectedAlert]);

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files?.length) return;

        setAnalyzing(true);
        const startTime = Date.now();
        const files = Array.from(e.target.files);

        try {
            const result = await api.analyzeCombined(files);
            const processingTime = ((Date.now() - startTime) / 1000).toFixed(2) + 's';
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
                rawResult: result, // Store raw result for detailed view if needed
                processingTime: processingTime
            };

            setAlerts([newAlert]);
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
            2. TECHNICAL FAULT CHARACTERISTICS: Observations from the visual spectrum analysis.
            3. RISK ASSESSMENT: Potential impact on train operations, track stability, and passenger safety.
            4. REMEDIAL ACTION PLAN: Immediate engineering steps (e.g., speed restrictions, weld repair).
            5. PREVENTATIVE RECOMMENDATIONS: Future measures to avoid recurrence.
            
            Use professional railway engineering terminology. 
            Total word count: 500-600 words.
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
                // If url refers to a local file or relative path that might fail, handle it
                if (url.startsWith('/')) {
                    img.src = window.location.origin + url;
                } else {
                    img.src = url;
                }
            });
        };

        try {
            const reportText = await generateGeminiResponse(prompt);
            const doc = new jsPDF();
            const pageWidth = doc.internal.pageSize.getWidth();
            const pageHeight = doc.internal.pageSize.getHeight();
            const margin = 15;
            const contentWidth = pageWidth - (margin * 2);
            let yPos = 15;

            // --- 0. Helper: Check Page & Add New ---
            const checkPageBreak = (neededSpace: number) => {
                if (yPos + neededSpace > pageHeight - margin) {
                    doc.addPage();
                    yPos = 20;
                    // Header on new page
                    doc.setFontSize(8);
                    doc.setTextColor(150);
                    doc.text(`Technical Incident Report: #INC-2026-${selectedAlert.id} (Continued)`, margin, 10);
                    doc.text(`Page ${doc.internal.pages.length - 1}`, pageWidth - margin, 10, { align: 'right' });
                }
            };

            // --- 1. Load Assets ---
            const irLogoData = await loadImage('/mock-images/Indian_Railways_Vector_Logo.png');
            const goiLogoData = await loadImage('/mock-images/Government_of_India_logo.svg');
            let incidentImageData = "";
            if (selectedAlert.img && !selectedAlert.img.includes('.csv') && selectedAlert.img !== 'csv_file') {
                incidentImageData = await loadImage(selectedAlert.img);
            }

            // Get Backend Charts (already base64)
            const timeSeriesChart = selectedAlert.rawResult?.result?.expert_results?.structural?.vibration_analysis?.statistics?.charts?.time_series;
            const heatmapChart = selectedAlert.rawResult?.result?.expert_results?.structural?.vibration_analysis?.statistics?.charts?.heatmap;

            // --- 2. Header ---
            // Govt Logo (Center Top)
            if (goiLogoData) {
                const logoW = 20;
                const logoH = 15;
                doc.addImage(goiLogoData, 'PNG', (pageWidth / 2) - (logoW / 2), yPos, logoW, logoH);
                yPos += 18;
            }

            doc.setFontSize(14);
            doc.setFont("helvetica", "bold");
            doc.setTextColor(0, 51, 102); // Navy Blue
            doc.text("INDIAN RAILWAYS - NORTHERN REGION", pageWidth / 2, yPos, { align: "center" });
            yPos += 7;
            doc.setFontSize(10);
            doc.setTextColor(60);
            doc.text("OFFICE OF THE CHIEF SAFETY OFFICER", pageWidth / 2, yPos, { align: "center" });
            yPos += 10;

            doc.setDrawColor(0, 51, 102);
            doc.setLineWidth(1.0);
            doc.line(margin, yPos, pageWidth - margin, yPos);
            yPos += 5;

            // --- 3. Incident Metadata Box ---
            doc.setFillColor(245, 247, 250);
            doc.rect(margin, yPos, contentWidth, 25, 'F');
            doc.setDrawColor(200);
            doc.rect(margin, yPos, contentWidth, 25, 'S');

            doc.setFont("helvetica", "bold");
            doc.setFontSize(11);
            doc.setTextColor(0);
            doc.text(`INCIDENT REPORT: #INC-${selectedAlert.id}`, margin + 5, yPos + 8);

            doc.setFontSize(9);
            doc.setFont("helvetica", "normal");
            doc.text(`Date/Time: ${new Date().toLocaleString()}`, margin + 5, yPos + 18);

            doc.setFont("helvetica", "bold");
            doc.text(`Severity:`, margin + 80, yPos + 8);
            doc.setTextColor(selectedAlert.severity === 'Critical' ? 200 : 0, 0, 0); // Red if Critical
            doc.text(selectedAlert.severity.toUpperCase(), margin + 98, yPos + 8);

            doc.setTextColor(0);
            doc.text(`Location:`, margin + 80, yPos + 18);
            doc.setFont("helvetica", "normal");
            doc.text(selectedAlert.location, margin + 98, yPos + 18);

            yPos += 35;

            // --- 4. Executive Summary & Analysis (Structured Parsing) ---
            // Parse the AI response into sections based on headers like "1. EXECUTIVE SUMMARY"
            const sections = [
                { title: "1. EXECUTIVE SUMMARY & ASSESSMENT", content: "" },
                { title: "2. TECHNICAL FAULT CHARACTERISTICS", content: "" },
                { title: "3. RISK ASSESSMENT", content: "" },
                { title: "4. REMEDIAL ACTION PLAN", content: "" },
                { title: "5. PREVENTATIVE RECOMMENDATIONS", content: "" }
            ];

            // Normalize text: remove markdown bold/headers to avoid artifacts, normalize newlines
            let cleanRawText = (reportText || "")
                .replace(/\*\*/g, "")
                .replace(/##/g, "")
                .replace(/\r\n/g, "\n");

            // Simple parser: split by title keywords
            // We'll iterate and find the text between titles
            let remainingText = cleanRawText;

            // Extract content for each section
            for (let i = 0; i < sections.length; i++) {
                const currentTitle = sections[i].title.split(':')[0]; // e.g., "1. EXECUTIVE SUMMARY"
                const nextTitle = sections[i + 1]?.title.split(':')[0];

                // Find start index search ignoring case/exact format
                // We typically expect the AI to generate "1. EXECUTIVE SUMMARY: text..."
                // But sometimes it puts the title on its own line.
                // We'll just take chunks sequentially if regex matches fail, or rely on the prompt structure.

                // Robust fallback approach: The prompt is very specific. 
                // Let's assume the AI produced the sections in order.
                // We will rely on finding the Section Headers in the text.

                const headerRegex = new RegExp(`${i + 1}\\.\\s*[A-Z ]+(:|\\n)`, 'i');
                const match = remainingText.match(headerRegex);

                if (match && match.index !== undefined) {
                    // Content starts after this header
                    let contentStart = match.index + match[0].length;

                    // Start searching for NEXT header from here
                    let contentEnd = remainingText.length;
                    if (nextTitle) {
                        const nextHeaderRegex = new RegExp(`${i + 2}\\.\\s*[A-Z ]+(:|\\n)`, 'i');
                        const nextMatch = remainingText.slice(contentStart).match(nextHeaderRegex);
                        if (nextMatch && nextMatch.index !== undefined) {
                            contentEnd = contentStart + nextMatch.index;
                        }
                    }

                    let sectionContent = remainingText.slice(contentStart, contentEnd).trim();
                    sections[i].content = sectionContent;

                    // Don't consume text, as we sliced from the original or logical start? 
                    // Actually easier to just regex match all.
                }
            }

            // Fallback: If parsing failed (empty sections), just dump the text in the first section
            if (!sections.some(s => s.content.length > 10)) {
                sections[0].content = cleanRawText;
            }

            // Render Sections
            sections.forEach(section => {
                if (!section.content) return;

                checkPageBreak(25);

                // Section Title (Bold, Colored)
                doc.setFont("helvetica", "bold");
                doc.setFontSize(12);
                doc.setTextColor(0, 51, 102);
                doc.text(section.title.split(':')[0], margin, yPos); // Print just "1. EXECUTIVE SUMMARY"
                yPos += 7;

                // Section Body (Normal, Black)
                doc.setFont("helvetica", "normal");
                doc.setFontSize(10);
                doc.setTextColor(40);

                const bodyLines = doc.splitTextToSize(section.content, contentWidth);
                bodyLines.forEach((line: string) => {
                    checkPageBreak(5);
                    doc.text(line, margin, yPos);
                    yPos += 5;
                });
                yPos += 8; // Spacing after section
            });

            // --- 5. Vibration Analysis Charts (If Available) ---
            if (timeSeriesChart || heatmapChart) {
                checkPageBreak(130); // Check largely if we have space for at least one chart + header

                doc.setFont("helvetica", "bold");
                doc.setFontSize(12);
                doc.setTextColor(0, 51, 102);
                doc.text("2. VIBRATION SIGNAL ANALYSIS", margin, yPos);
                yPos += 8;

                if (timeSeriesChart) {
                    checkPageBreak(70);
                    doc.addImage(`data:image/png;base64,${timeSeriesChart}`, 'PNG', margin, yPos, contentWidth, 60);
                    doc.setFontSize(8);
                    doc.setTextColor(100);
                    doc.text("Fig 1: 3-Axis Vibration Acceleration Time Series", margin, yPos + 65);
                    yPos += 75;
                }

                if (heatmapChart) {
                    checkPageBreak(50);
                    doc.addImage(`data:image/png;base64,${heatmapChart}`, 'PNG', margin, yPos, contentWidth, 30);
                    doc.setFontSize(8);
                    doc.setTextColor(100);
                    doc.text("Fig 2: Vibration Intensity Heatmap (Spectral Density)", margin, yPos + 35);
                    yPos += 45;
                }
            }

            // --- 6. Remedial Action Plan (Table) ---
            const actions = selectedAlert.rawResult?.result?.action_report?.recommended_actions;
            if (actions && actions.length > 0) {
                checkPageBreak(60);

                doc.setFont("helvetica", "bold");
                doc.setFontSize(12);
                doc.setTextColor(0, 51, 102);
                doc.text("3. REMEDIAL ACTION PLAN", margin, yPos);
                yPos += 8;

                // Table Header
                doc.setFillColor(240, 240, 240);
                doc.rect(margin, yPos, contentWidth, 8, 'F');
                doc.setFontSize(9);
                doc.setTextColor(0);
                doc.text("Action Item", margin + 2, yPos + 6);
                doc.text("Urgency", margin + 90, yPos + 6);
                doc.text("Owner", margin + 120, yPos + 6);
                yPos += 10;

                // Table Rows
                doc.setFont("helvetica", "normal");
                actions.forEach((action: any, i: number) => {
                    checkPageBreak(15);
                    const actionText = doc.splitTextToSize(action.action, 85);
                    const rowHeight = Math.max(actionText.length * 4 + 4, 10);

                    // Stripe background
                    if (i % 2 === 1) {
                        doc.setFillColor(252, 252, 252);
                        doc.rect(margin, yPos - 1, contentWidth, rowHeight, 'F');
                    }

                    doc.text(actionText, margin + 2, yPos + 3);

                    // Urgency Badge-like text
                    doc.setFont("helvetica", "bold");
                    const urgency = action.urgency || "ASAP";
                    if (urgency.includes("T+0")) doc.setTextColor(200, 0, 0);
                    else doc.setTextColor(0);
                    doc.text(urgency, margin + 90, yPos + 3);

                    doc.setTextColor(0);
                    doc.setFont("helvetica", "normal");
                    doc.text(action.owner || "Engineering", margin + 120, yPos + 3);

                    yPos += rowHeight;
                });
                yPos += 10;
            }

            // --- 7. Signature / Footer ---
            checkPageBreak(30);
            doc.setDrawColor(150);
            doc.setLineWidth(0.5);
            doc.line(margin, yPos, pageWidth / 2, yPos);
            yPos += 5;
            doc.setFontSize(8);
            doc.setFont("helvetica", "bold");
            doc.text("SYSTEM GENERATED REPORT", margin, yPos);
            doc.setFont("helvetica", "normal");
            doc.text("Automated Railway Safety Oversight Drone (RSOD) System", margin, yPos + 4);
            doc.text("This document is electronically verified.", margin, yPos + 8);

            doc.save(`IR_Safety_Report_${selectedAlert.id}.pdf`);

        } catch (e) {
            console.error(e);
            alert("Error generating report: " + e);
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

                    {/* Detail Content - Enhanced with full data display */}
                    <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(100vh-300px)]">
                        {/* Top Grid: Media + Summary */}
                        <div className="grid grid-cols-2 gap-6">
                            {/* Left: Media Display */}
                            <div className="space-y-4">
                                <div className="relative rounded-lg overflow-hidden border border-gray-300 group bg-gray-100 flex items-center justify-center min-h-[200px]">
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
                                        <div className="relative">
                                            <video src={selectedAlert.img} controls autoPlay muted loop className="w-full h-auto max-h-[400px] object-contain rounded">
                                                Your browser does not support the video tag.
                                            </video>
                                            <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur">
                                                Confidence: {selectedAlert.score}%
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="relative">
                                            <Image src={selectedAlert.img} alt="Evidence" width={500} height={300} className="w-full h-auto object-cover" unoptimized />
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
                                        <p className="font-mono font-bold text-xs text-gray-700">{selectedAlert.processingTime || '~2.5s'}</p>
                                    </div>
                                </div>
                            </div>

                            {/* Right: Analysis Summary */}
                            <div className="space-y-4">
                                <div>
                                    <h4 className="font-bold text-gray-800 mb-2">AI Analysis</h4>
                                    <p className="text-sm text-gray-600 leading-relaxed">
                                        {selectedAlert.rawResult?.result?.expert_results?.structural ? (
                                            <>Structural vibration analysis detected a <strong>{selectedAlert.type}</strong> at {selectedAlert.location}.
                                                {selectedAlert.rawResult.result.expert_results.structural.vibration_analysis?.failure_ratio > 0 &&
                                                    ` ${(selectedAlert.rawResult.result.expert_results.structural.vibration_analysis.failure_ratio * 100).toFixed(1)}% of analyzed windows show failure patterns.`}
                                            </>
                                        ) : (
                                            <>The system detected a potential <strong>{selectedAlert.type}</strong> at {selectedAlert.location}.
                                                Visual spectrum analysis indicates structural anomaly consistent with {selectedAlert.type.toLowerCase()}.</>
                                        )}
                                        {' '}Immediate inspection recommended.
                                    </p>
                                </div>

                                {/* Detection List */}
                                {selectedAlert.detections && selectedAlert.detections.length > 0 && (
                                    <div className="bg-gray-50 p-3 rounded border border-gray-200">
                                        <h5 className="font-bold text-xs text-gray-700 mb-2">Detected Objects ({selectedAlert.detections.length})</h5>
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

                                {/* Structural Alerts */}
                                {selectedAlert.rawResult?.result?.expert_results?.structural?.alerts && (
                                    <div className="bg-red-50 p-3 rounded border border-red-200">
                                        <h5 className="font-bold text-xs text-red-800 mb-2">Structural Alerts</h5>
                                        <ul className="text-xs text-red-700 space-y-1">
                                            {selectedAlert.rawResult.result.expert_results.structural.alerts.map((alert: string, idx: number) => (
                                                <li key={idx}>{alert}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Visual Alerts */}
                                {selectedAlert.rawResult?.result?.expert_results?.visual?.alerts && (
                                    <div className="bg-red-50 p-3 rounded border border-red-200">
                                        <h5 className="font-bold text-xs text-red-800 mb-2">Visual Alerts</h5>
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
                            </div>
                        </div>

                        {/* Vibration Statistics - Only show for structural data */}
                        {selectedAlert.rawResult?.result?.expert_results?.structural?.vibration_analysis?.statistics && (
                            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
                                <h4 className="font-bold text-gray-800 mb-3 flex items-center gap-2">
                                    Vibration Statistics
                                </h4>
                                <div className="grid grid-cols-3 gap-4">
                                    {['x', 'y', 'z'].map((axis) => {
                                        const stats = selectedAlert.rawResult.result.expert_results.structural.vibration_analysis.statistics[axis];
                                        return (
                                            <div key={axis} className="bg-white p-3 rounded shadow-sm">
                                                <h5 className="font-bold text-xs text-gray-600 mb-2 uppercase">{axis}-Axis</h5>
                                                <div className="space-y-1 text-xs">
                                                    <div className="flex justify-between">
                                                        <span className="text-gray-500">Mean:</span>
                                                        <span className="font-mono font-bold">{stats.mean?.toFixed(3) || 'N/A'}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-gray-500">Median:</span>
                                                        <span className="font-mono font-bold">{stats.median?.toFixed(3) || 'N/A'}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-gray-500">RMS:</span>
                                                        <span className="font-mono font-bold">{stats.rms?.toFixed(3) || 'N/A'}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-gray-500">Std Dev:</span>
                                                        <span className="font-mono font-bold">{stats.std?.toFixed(3) || 'N/A'}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-gray-500">Range:</span>
                                                        <span className="font-mono font-bold text-xs">[{stats.min?.toFixed(1)}, {stats.max?.toFixed(1)}]</span>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Vibration Plots - Matplotlib generated images */}
                        {selectedAlert.rawResult?.result?.expert_results?.structural?.vibration_analysis?.statistics?.charts?.time_series && (
                            <div className="bg-white p-4 rounded-lg border border-gray-200">
                                <h4 className="font-bold text-gray-800 mb-3">Vibration Time Series</h4>
                                <img
                                    src={`data:image/png;base64,${selectedAlert.rawResult.result.expert_results.structural.vibration_analysis.statistics.charts.time_series}`}
                                    alt="Vibration Time Series"
                                    className="w-full h-auto rounded"
                                />
                            </div>
                        )}

                        {/* Vibration Plots - Simple SVG-based charts (Fallback) */}
                        {!selectedAlert.rawResult?.result?.expert_results?.structural?.vibration_analysis?.statistics?.charts?.time_series && selectedAlert.rawResult?.result?.expert_results?.structural?.vibration_analysis?.statistics?.sampled_data && (
                            <div className="bg-white p-4 rounded-lg border border-gray-200">
                                <h4 className="font-bold text-gray-800 mb-3">Vibration Time Series</h4>
                                <div className="space-y-4">
                                    {['x', 'y', 'z'].map((axis) => {
                                        const sampledData = selectedAlert.rawResult.result.expert_results.structural.vibration_analysis.statistics.sampled_data;
                                        const data = sampledData[axis] || [];
                                        const stats = selectedAlert.rawResult.result.expert_results.structural.vibration_analysis.statistics[axis];

                                        if (!data.length) return null;

                                        const max = Math.max(...data);
                                        const min = Math.min(...data);
                                        const range = max - min || 1;
                                        const width = 600;
                                        const height = 80;
                                        const padding = 10;

                                        const points = data.map((val: number, idx: number) => {
                                            const x = padding + (idx / (data.length - 1)) * (width - 2 * padding);
                                            const y = height - padding - ((val - min) / range) * (height - 2 * padding);
                                            return `${x},${y}`;
                                        }).join(' ');

                                        return (
                                            <div key={axis} className="border border-gray-200 rounded p-2">
                                                <div className="flex justify-between items-center mb-1">
                                                    <span className="text-xs font-bold text-gray-700 uppercase">{axis}-Axis</span>
                                                    <div className="flex gap-3 text-xs">
                                                        <span className="text-blue-600">Mean: {stats.mean?.toFixed(2)}</span>
                                                        <span className="text-green-600">Median: {stats.median?.toFixed(2)}</span>
                                                        <span className="text-purple-600">RMS: {stats.rms?.toFixed(2)}</span>
                                                    </div>
                                                </div>
                                                <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="bg-gray-50 rounded">
                                                    <polyline points={points} fill="none" stroke="#3b82f6" strokeWidth="1.5" />
                                                    <line x1={padding} y1={height - padding - ((stats.mean - min) / range) * (height - 2 * padding)}
                                                        x2={width - padding} y2={height - padding - ((stats.mean - min) / range) * (height - 2 * padding)}
                                                        stroke="#3b82f6" strokeWidth="1" strokeDasharray="4,4" opacity="0.5" />
                                                </svg>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}


                        {/* Heatmap Visualization */}
                        {selectedAlert.rawResult?.result?.expert_results?.structural?.vibration_analysis?.statistics?.charts?.heatmap ? (
                            <div className="bg-white p-4 rounded-lg border border-gray-200">
                                <h4 className="font-bold text-gray-800 mb-3">Vibration Heatmap (3-Axis)</h4>
                                <img
                                    src={`data:image/png;base64,${selectedAlert.rawResult.result.expert_results.structural.vibration_analysis.statistics.charts.heatmap}`}
                                    alt="Vibration Heatmap"
                                    className="w-full h-auto rounded"
                                />
                            </div>
                        ) : selectedAlert.rawResult?.result?.expert_results?.structural?.vibration_analysis?.statistics?.sampled_data && (
                            <div className="bg-white p-4 rounded-lg border border-gray-200">
                                <h4 className="font-bold text-gray-800 mb-3">Vibration Heatmap (3-Axis)</h4>
                                <div className="overflow-x-auto">
                                    <svg width="600" height="120" viewBox="0 0 600 120" className="bg-gray-50 rounded">
                                        {['x', 'y', 'z'].map((axis, axisIdx) => {
                                            const sampledData = selectedAlert.rawResult.result.expert_results.structural.vibration_analysis.statistics.sampled_data;
                                            const data = sampledData[axis] || [];
                                            const max = Math.max(...data);
                                            const min = Math.min(...data);
                                            const range = max - min || 1;

                                            return data.slice(0, 100).map((val: number, idx: number) => {
                                                const intensity = (val - min) / range;
                                                const color = intensity > 0.7 ? '#ef4444' : intensity > 0.4 ? '#f97316' : intensity > 0.2 ? '#fbbf24' : '#3b82f6';
                                                return (
                                                    <rect
                                                        key={`${axis}-${idx}`}
                                                        x={idx * 6}
                                                        y={axisIdx * 35 + 10}
                                                        width="5"
                                                        height="30"
                                                        fill={color}
                                                        opacity={0.7 + intensity * 0.3}
                                                    />
                                                );
                                            });
                                        })}
                                        <text x="10" y="30" fontSize="10" fill="#666">X</text>
                                        <text x="10" y="65" fontSize="10" fill="#666">Y</text>
                                        <text x="10" y="100" fontSize="10" fill="#666">Z</text>
                                    </svg>
                                </div>
                                <div className="flex items-center gap-2 mt-2 text-xs text-gray-600">
                                    <span>Intensity:</span>
                                    <div className="flex gap-1">
                                        <div className="w-4 h-4 bg-blue-500 rounded"></div>
                                        <span>Low</span>
                                    </div>
                                    <div className="flex gap-1">
                                        <div className="w-4 h-4 bg-yellow-400 rounded"></div>
                                        <span>Med</span>
                                    </div>
                                    <div className="flex gap-1">
                                        <div className="w-4 h-4 bg-red-500 rounded"></div>
                                        <span>High</span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Action Plan Table */}
                        {selectedAlert.rawResult?.result?.action_report?.recommended_actions && (
                            <div className="bg-white p-4 rounded-lg border border-gray-200">
                                <h4 className="font-bold text-gray-800 mb-3">Recommended Actions</h4>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-xs">
                                        <thead className="bg-gray-100">
                                            <tr>
                                                <th className="text-left p-2 font-bold text-gray-700">Action</th>
                                                <th className="text-left p-2 font-bold text-gray-700">Owner</th>
                                                <th className="text-left p-2 font-bold text-gray-700">Urgency</th>
                                                <th className="text-left p-2 font-bold text-gray-700">Impact</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {selectedAlert.rawResult.result.action_report.recommended_actions.map((action: any, idx: number) => (
                                                <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                                                    <td className="p-2">{action.action}</td>
                                                    <td className="p-2 font-medium text-blue-600">{action.owner}</td>
                                                    <td className="p-2">
                                                        <span className={`px-2 py-0.5 rounded text-xs font-bold ${action.urgency?.includes('T+0') ? 'bg-red-100 text-red-700' :
                                                            action.urgency?.includes('T+5') ? 'bg-orange-100 text-orange-700' :
                                                                'bg-yellow-100 text-yellow-700'
                                                            }`}>
                                                            {action.urgency}
                                                        </span>
                                                    </td>
                                                    <td className="p-2 text-gray-600">{action.impact}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}

                        {/* Root Cause Analysis */}
                        {selectedAlert.rawResult?.result?.tampering_analysis?.root_cause_analysis && (
                            <div className="bg-amber-50 p-4 rounded-lg border border-amber-200">
                                <h4 className="font-bold text-gray-800 mb-3">Root Cause Analysis</h4>
                                <div className="space-y-2 text-sm">
                                    <div>
                                        <span className="font-bold text-gray-700">Probable Cause: </span>
                                        <span className="text-gray-600">{selectedAlert.rawResult.result.tampering_analysis.root_cause_analysis.probable_cause}</span>
                                    </div>
                                    {selectedAlert.rawResult.result.tampering_analysis.root_cause_analysis.contributing_factors?.length > 0 && (
                                        <div>
                                            <span className="font-bold text-gray-700">Contributing Factors:</span>
                                            <ul className="list-disc list-inside text-gray-600 ml-2">
                                                {selectedAlert.rawResult.result.tampering_analysis.root_cause_analysis.contributing_factors.map((factor: string, idx: number) => (
                                                    <li key={idx}>{factor}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {selectedAlert.rawResult.result.tampering_analysis.tampering_assessment?.evidence?.length > 0 && (
                                        <div>
                                            <span className="font-bold text-gray-700">Evidence:</span>
                                            <ul className="list-disc list-inside text-gray-600 ml-2">
                                                {selectedAlert.rawResult.result.tampering_analysis.tampering_assessment.evidence.map((ev: string, idx: number) => (
                                                    <li key={idx}>{ev}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Recommendations */}
                        {(selectedAlert.rawResult?.result?.expert_results?.structural?.recommendations ||
                            selectedAlert.rawResult?.result?.expert_results?.visual?.recommendations) && (
                                <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                                    <h4 className="font-bold text-yellow-800 mb-2">Expert Recommendations</h4>
                                    <ul className="text-sm text-yellow-700 space-y-1">
                                        {selectedAlert.rawResult.result.expert_results.structural?.recommendations?.map((rec: string, idx: number) => (
                                            <li key={`s-${idx}`}>• {rec}</li>
                                        ))}
                                        {selectedAlert.rawResult.result.expert_results.visual?.recommendations?.map((rec: string, idx: number) => (
                                            <li key={`v-${idx}`}>• {rec}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                    </div>
                </div>
            </div>
        </div>
    );
}
