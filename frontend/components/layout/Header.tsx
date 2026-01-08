import { Shield } from 'lucide-react';

export default function Header() {
    return (
        <header className="bg-white border-b border-govt-border h-20 flex items-center px-6 sticky top-0 z-50 shadow-sm">
            <div className="flex items-center gap-4">
                {/* Official Logos */}
                <div className="flex items-center gap-4">
                    <img src="/assets/govt_logo.svg" alt="Govt of India" className="h-12 w-auto" />
                    <div className="h-10 w-[1px] bg-gray-300"></div>
                    <img src="/assets/railway_logo.png" alt="Indian Railways" className="h-12 w-auto" />
                </div>

                <div className="flex flex-col ml-4">
                    <h1 className="text-xl font-bold text-govt-navy leading-tight">Indian Railways</h1>
                    <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Government of India</span>
                </div>
            </div>

            <div className="ml-8 border-l border-gray-300 pl-8 h-10 flex flex-col justify-center">
                <h2 className="text-lg font-semibold text-gray-800">Railway Tampering Detection System</h2>
            </div>

            <div className="ml-auto flex items-center gap-4">
                <div className="text-right hidden md:block">
                    <p className="text-sm font-medium text-gray-900">Officer A. Sharma</p>
                    <p className="text-xs text-gray-500">Zone Engineer - North</p>
                </div>
                <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center text-gray-600 font-bold">
                    AS
                </div>
            </div>
        </header>
    );
}
