"""
Email Sender Module for CareerMatch AI
Handles sending job recommendation results via email
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import streamlit as st
from typing import Dict, List
import pandas as pd


def get_email_config() -> Dict[str, str]:
    try:
        return {
            'smtp_server': st.secrets['email']['smtp_server'],
            'smtp_port': int(st.secrets['email']['smtp_port']),
            'sender_email': st.secrets['email']['sender_email'],
            'sender_password': st.secrets['email']['sender_password'],
            'sender_name': st.secrets['email']['sender_name']
        }
    except Exception as e:
        raise Exception(
            "Email belum dikonfigurasi. "
            "Silakan setup secrets mengikuti panduan di EMAIL_SETUP.md"
        ) from e


def load_email_template() -> str:
    template_path = Path(__file__).parent / "email_template.html"
    
    if not template_path.exists():
        return """
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>Hasil Rekomendasi Karir - CareerMatch AI</h2>
            {content}
        </body>
        </html>
        """
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_email_html(
    recipient_email: str,
    top_job: Dict,
    results_df: pd.DataFrame,
    avg_score: float,
    tech_score: float,
    soft_score: float,
    key_drivers: List[Dict]
) -> str:
    template = load_email_template()
    
    drivers_html = ""
    if key_drivers:
        for driver in key_drivers[:4]:
            drivers_html += f"""
            <div style="background: #f8fafc; padding: 12px; border-radius: 8px; text-align: center; margin: 0 5px;">
                <div style="font-size: 24px; margin-bottom: 5px;">⭐</div>
                <div style="font-weight: 700; color: #6366f1; font-size: 13px; margin-bottom: 5px;">
                    {driver['feature']}
                </div>
                <div style="font-size: 12px; color: #64748b; background: #e0e7ff; padding: 3px 10px; border-radius: 12px;">
                    Skor: {driver['original_score']}
                </div>
            </div>
            """
    
    leaderboard_html = ""
    for _, row in results_df.head(10).iterrows():
        rank = row['Rank']
        medal = ""
        if rank == 1:
            medal = "🥇"
        elif rank == 2:
            medal = "🥈"
        elif rank == 3:
            medal = "🥉"
        
        leaderboard_html += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; font-weight: 600; color: #6b7280;">
                {rank}
            </td>
            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #1f2937;">
                {medal} {row['Job Role']}
            </td>
            <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: right;">
                <span style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; padding: 4px 12px; border-radius: 50px; font-size: 12px; font-weight: 600;">
                    {row['Match Score']}
                </span>
            </td>
        </tr>
        """
    
    html_content = template.format(
        recipient_email=recipient_email,
        top_job_role=top_job['Job Role'],
        top_job_score=top_job['Match Score'],
        avg_score=f"{avg_score:.1f}",
        tech_score=f"{tech_score:.1f}",
        soft_score=f"{soft_score:.1f}",
        key_drivers=drivers_html if drivers_html else "<p style='text-align: center; color: #64748b;'>Profil Anda memiliki keseimbangan skill yang unik</p>",
        leaderboard=leaderboard_html
    )
    
    return html_content


def send_recommendation_email(
    recipient_email: str,
    top_job: Dict,
    results_df: pd.DataFrame,
    avg_score: float,
    tech_score: float,
    soft_score: float,
    key_drivers: List[Dict]
) -> tuple[bool, str]:
    try:
        config = get_email_config()
        
        message = MIMEMultipart('alternative')
        message['Subject'] = f"Hasil Rekomendasi Karir - {top_job['Job Role']}"
        message['From'] = f"{config['sender_name']} <{config['sender_email']}>"
        message['To'] = recipient_email
        
        html_content = generate_email_html(
            recipient_email,
            top_job,
            results_df,
            avg_score,
            tech_score,
            soft_score,
            key_drivers
        )
        
        html_part = MIMEText(html_content, 'html', 'utf-8')
        message.attach(html_part)
        
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['sender_email'], config['sender_password'])
            server.send_message(message)
        
        return True, f"✅ Email berhasil dikirim ke {recipient_email}"
    
    except Exception as e:
        error_msg = str(e)
        
        if "Authentication" in error_msg or "Username and Password not accepted" in error_msg:
            return False, "❌ Autentikasi gagal. Periksa email dan password di konfigurasi."
        elif "Connection" in error_msg or "timed out" in error_msg:
            return False, "❌ Koneksi ke email server gagal. Periksa koneksi internet Anda."
        elif "secrets" in error_msg.lower():
            return False, "❌ Email belum dikonfigurasi. Hubungi administrator."
        else:
            return False, f"❌ Gagal mengirim email: {error_msg}"
