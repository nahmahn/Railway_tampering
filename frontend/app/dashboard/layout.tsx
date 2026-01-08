import Header from '@/components/layout/Header';
import Sidebar from '@/components/layout/Sidebar';

export default function DashboardLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <div className="min-h-screen bg-gray-50 flex flex-col font-sans">
            <Header />
            <div className="flex flex-1">
                <Sidebar />
                <main className="flex-1 p-8 overflow-y-auto h-[calc(100vh-80px)]">
                    {children}
                </main>
            </div>
        </div>
    );
}
