import "./globals.css";
import React from "react";

export const metadata = {
    title: "Human Evaluation App",
    description: "MVP annotation app for multimodal prediction verification"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    );
}
