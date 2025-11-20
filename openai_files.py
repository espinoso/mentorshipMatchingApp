"""
OpenAI File Management functions for the Mentorship Matching System
Contains functions for listing, uploading, and deleting files in OpenAI storage
"""

from openai import OpenAI
import streamlit as st
from io import BytesIO
from datetime import datetime


def list_openai_files(api_key):
    """List all files uploaded to OpenAI storage"""
    try:
        client = OpenAI(api_key=api_key)
        files = client.files.list(purpose='assistants')
        
        file_list = []
        for file in files.data:
            created_at = datetime.fromtimestamp(file.created_at)
            age_days = (datetime.now() - created_at).days
            
            file_list.append({
                'id': file.id,
                'filename': file.filename,
                'size_bytes': file.bytes,
                'size_kb': file.bytes / 1024,
                'created_at': created_at,
                'age_days': age_days,
                'purpose': file.purpose
            })
        
        return file_list
    except Exception as e:
        st.error(f"❌ Error listing files: {str(e)}")
        return []


def upload_training_files_to_openai(training_dfs, api_key):
    """Upload training DataFrames to OpenAI storage"""
    try:
        client = OpenAI(api_key=api_key)
        uploaded_file_ids = []
        
        for i, df in enumerate(training_dfs):
            csv_content = df.to_csv(index=False)
            
            file_obj = BytesIO(csv_content.encode('utf-8'))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_obj.name = f"training_data_{timestamp}_{i+1}.csv"
            
            st.info(f"📤 Uploading {file_obj.name} ({len(csv_content)/1024:.1f} KB)...")
            
            file = client.files.create(
                file=file_obj,
                purpose='assistants'
            )
            
            uploaded_file_ids.append(file.id)
            st.success(f"✅ Uploaded: {file.id}")
        
        return uploaded_file_ids
        
    except Exception as e:
        st.error(f"❌ Error uploading files: {str(e)}")
        return []


def delete_openai_files(api_key, file_ids):
    """Delete specific files from OpenAI storage"""
    try:
        client = OpenAI(api_key=api_key)
        deleted_count = 0
        
        for file_id in file_ids:
            try:
                client.files.delete(file_id)
                deleted_count += 1
            except Exception as e:
                st.warning(f"Could not delete {file_id}: {str(e)}")
        
        return deleted_count
        
    except Exception as e:
        st.error(f"❌ Error deleting files: {str(e)}")
        return 0

