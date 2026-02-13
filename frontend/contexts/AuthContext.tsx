"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { googleLogout } from "@react-oauth/google";

interface User {
    name?: string;
    email?: string;
    picture?: string;
    token?: string;
}

interface AuthContextType {
    user: User | null;
    login: (token: string, userData?: any) => void;
    logout: () => void;
    loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        // Check for stored token on mount
        const storedToken = localStorage.getItem("google_token");
        const storedUser = localStorage.getItem("user_data");

        if (storedToken) {
            try {
                const userData = storedUser ? JSON.parse(storedUser) : {};
                setUser({ ...userData, token: storedToken });
            } catch (e) {
                console.error("Failed to parse user data", e);
                localStorage.removeItem("google_token");
            }
        }
        setLoading(false);
    }, []);

    const login = (token: string, userData: any) => {
        localStorage.setItem("google_token", token);
        localStorage.setItem("user_data", JSON.stringify(userData));
        setUser({ ...userData, token });
        router.push("/dashboard/overview");
    };

    const logout = () => {
        googleLogout();
        localStorage.removeItem("google_token");
        localStorage.removeItem("user_data");
        setUser(null);
        router.push("/login");
    };

    return (
        <AuthContext.Provider value={{ user, login, logout, loading }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
}
