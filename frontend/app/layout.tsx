import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Railway Tampering Detection System",
  description: "Official Monitoring Portal - Govt of India",
};

import { GoogleOAuthProvider } from '@react-oauth/google';
import { AlertProvider } from '@/contexts/AlertContext';
import { AuthProvider } from '@/contexts/AuthContext';

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} antialiased`}>
        <GoogleOAuthProvider clientId="741472179168-ujrglefc9vjcusv1pg0muqhqihavds12.apps.googleusercontent.com">
          <AuthProvider>
            <AlertProvider>
              {children}
            </AlertProvider>
          </AuthProvider>
        </GoogleOAuthProvider>
      </body>
    </html>
  );
}

