export default function DatasetModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4" onClick={onClose}>
            <div className="bg-white rounded-lg overflow-hidden max-w-4xl w-full max-h-[90vh] flex flex-col">
                <div className="p-4 border-b flex justify-between items-center bg-gray-50">
                    <h3 className="font-bold text-lg text-govt-navy">Detailed Dataset Visualization</h3>
                    <button onClick={onClose} className="text-gray-500 hover:text-red-500">Close</button>
                </div>
                <div className="overflow-auto p-4 flex-1 bg-gray-100 flex justify-center">
                    <img src="/mock-images/dataset_vis.jpg" alt="Dataset Visualization" className="max-w-full h-auto shadow-lg" />
                </div>
            </div>
        </div>
    );
}
