
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';
const UPLOADS_BASE_URL = 'http://localhost:8000'; // Changed to root since path includes /uploads now

export interface AnalysisResult {
    success: boolean;
    session_id: string;
    timestamp: string;
    result: CombinedAnalysisData;
    alerts: Alert[];
}

export interface CombinedAnalysisData {
    visual_result?: any;
    thermal_result?: any;
    structural_result?: any;
    tampering_analysis?: TamperingAnalysis;
    structured_output?: any;
    overall_risk_level?: string;
    tampering_detected?: boolean;
    confidence?: number;
    expert_results?: {
        visual?: any;
        thermal?: any;
        structural?: any;
    };
    overall_assessment?: {
        risk_level?: string;
        tampering_detected?: boolean;
        confidence?: number;
    };
}

export interface TamperingAnalysis {
    tampering_assessment: {
        is_tampering_detected: boolean;
        tampering_type: string;
        confidence: number;
        evidence: string[];
    };
    severity_classification: {
        tier: number;
        description: string;
        immediate_risk: boolean;
    };
}

export interface Alert {
    message: string;
}

// Add auth token to requests
axios.interceptors.request.use((config) => {
    if (typeof window !== 'undefined') {
        const token = localStorage.getItem('google_token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
    }
    return config;
});

export const api = {
    analyzeCombined: async (
        files: File[],
        context: any = {}
    ): Promise<AnalysisResult> => {
        const formData = new FormData();

        files.forEach(file => {
            if (file.type.startsWith('image/')) {
                formData.append('images', file);
            } else if (file.type.startsWith('video/')) {
                formData.append('videos', file);
            } else if (file.name.endsWith('.npy') || file.name.endsWith('.pcd') || file.name.endsWith('.ply')) {
                formData.append('lidar_files', file);
            } else if (file.name.endsWith('.csv')) {
                formData.append('csv_files', file);
            }
        });

        formData.append('context', JSON.stringify(context));

        try {
            const response = await axios.post(`${API_BASE_URL}/analyze/combined`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            return response.data;
        } catch (error) {
            console.error("API Error:", error);
            throw error;
        }
    },

    getUploadUrl: (filename: string) => {
        if (!filename) return '/placeholder.png'; // Fallback
        if (filename.startsWith('http') || filename.startsWith('data:')) return filename;

        // Normalize Windows backslashes to forward slashes
        let cleanPath = filename.replace(/\\/g, '/');

        // Remove leading slash if present to avoid double slashes
        cleanPath = cleanPath.startsWith('/') ? cleanPath.slice(1) : cleanPath;

        // If the path already starts with 'uploads/', don't add it again
        if (cleanPath.startsWith('uploads/')) {
            return `${UPLOADS_BASE_URL}/${cleanPath}`;
        }

        // Otherwise, prepend /uploads/
        return `${UPLOADS_BASE_URL}/uploads/${cleanPath}`;
    },

    getHistory: async (): Promise<any[]> => {
        try {
            const response = await axios.get(`${API_BASE_URL}/history`);
            return response.data.history;
        } catch (error) {
            console.error("API Error (History):", error);
            return [];
        }
    },

    query: async (query: string, context: any = {}): Promise<string> => {
        try {
            const response = await axios.post(`${API_BASE_URL}/query`, {
                query,
                context
            });
            return response.data.result.response;
        } catch (error) {
            console.error("API Error (Query):", error);
            throw error;
        }
    },

    // ==================== Crews API ====================

    getCrews: async (): Promise<any[]> => {
        try {
            const response = await axios.get(`${API_BASE_URL}/crews`);
            return response.data.crews;
        } catch (error) {
            console.error("API Error (Crews):", error);
            return [];
        }
    },

    updateCrewStatus: async (crewId: string, status: string): Promise<boolean> => {
        try {
            const formData = new FormData();
            formData.append('status', status);
            await axios.patch(`${API_BASE_URL}/crews/${crewId}/status`, formData);
            return true;
        } catch (error) {
            console.error("API Error (Crew Status):", error);
            return false;
        }
    },

    // ==================== Missions API ====================

    getMissions: async (stage?: string): Promise<any[]> => {
        try {
            const params = stage ? { stage } : {};
            const response = await axios.get(`${API_BASE_URL}/missions`, { params });
            return response.data.missions;
        } catch (error) {
            console.error("API Error (Missions):", error);
            return [];
        }
    },

    createMission: async (alertSessionId: string, priority: string = 'P2', crewId?: string, notes: string = ''): Promise<any> => {
        try {
            const response = await axios.post(`${API_BASE_URL}/missions`, {
                alert_session_id: alertSessionId,
                priority,
                crew_id: crewId || null,
                notes
            });
            return response.data;
        } catch (error) {
            console.error("API Error (Create Mission):", error);
            throw error;
        }
    },

    advanceMission: async (missionId: string, crewId?: string, priority?: string): Promise<any> => {
        try {
            const response = await axios.patch(`${API_BASE_URL}/missions/${missionId}/advance`, {
                crew_id: crewId || null,
                priority: priority || null
            });
            return response.data;
        } catch (error) {
            console.error("API Error (Advance Mission):", error);
            throw error;
        }
    },

    resolveMission: async (missionId: string, resolutionNotes: string): Promise<any> => {
        try {
            const response = await axios.patch(`${API_BASE_URL}/missions/${missionId}/resolve`, {
                resolution_notes: resolutionNotes
            });
            return response.data;
        } catch (error) {
            console.error("API Error (Resolve Mission):", error);
            throw error;
        }
    }
};
