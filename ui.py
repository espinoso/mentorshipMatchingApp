CSS_STYLES = """
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Improved main header with animation */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: fadeInDown 0.6s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main-header h1 {
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        font-weight: 300;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Sticky header */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: white;
        padding: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: -1rem -1rem 1rem -1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Enhanced mentee card */
    .mentee-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 0.6rem 0;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }
    
    .mentee-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: 0;
    }
    
    .mentee-card:hover {
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
        border-left-width: 6px;
    }
    
    .mentee-card.selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-left: 6px solid #5a6fd8 !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4) !important;
        transform: translateX(12px) scale(1.03);
    }
    
    .mentee-card * {
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced match card */
    .match-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
        padding: 1.8rem;
        border-radius: 16px;
        border: 2px solid #e9ecef;
        margin: 1.2rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        color: #212529 !important;
        transition: all 0.3s ease;
        animation: slideInRight 0.4s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .match-card:hover {
        box-shadow: 0 8px 28px rgba(0,0,0,0.12);
        transform: translateY(-4px);
        border-color: #667eea;
    }
    
    .match-card h4 {
        color: #2d3748 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.01em !important;
    }
    
    .match-card p {
        color: #4a5568 !important;
        margin-bottom: 0.75rem !important;
        line-height: 1.7 !important;
        font-size: 0.95rem !important;
    }
    
    .match-card strong {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    /* Enhanced percentage badge with animation */
    .percentage-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 24px;
        font-weight: 600;
        color: white;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .excellent { 
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    .strong { 
        background: linear-gradient(135deg, #20c997 0%, #17a2b8 100%);
    }
    .good { 
        background: linear-gradient(135deg, #17a2b8 0%, #20c997 100%);
    }
    .fair { 
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: #212529;
    }
    
    /* Enhanced alert boxes */
    .conflict-warning {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15);
        animation: shake 0.5s ease;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    .cost-estimate {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
    }
    
    .token-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        color: #856404;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15);
        animation: bounce 1s ease;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Wizard steps */
    .wizard-step {
        display: flex;
        align-items: center;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: #f8f9fa;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .wizard-step.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #5a6fd8;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .wizard-step.completed {
        background: #d4edda;
        border-color: #28a745;
    }
    
    .wizard-step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #667eea;
        color: white;
        font-weight: 600;
        margin-right: 0.8rem;
        font-size: 0.9rem;
    }
    
    .wizard-step.completed .wizard-step-number {
        background: #28a745;
    }
    
    .wizard-step.active .wizard-step-number {
        background: white;
        color: #667eea;
    }
    
    /* Info card */
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.15);
    }
    
    /* Success card */
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.15);
        animation: fadeInUp 0.5s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Loading skeleton */
    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: 8px;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Tooltip */
    .tooltip-icon {
        display: inline-block;
        width: 18px;
        height: 18px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 18px;
        font-size: 12px;
        font-weight: bold;
        cursor: help;
        margin-left: 0.3rem;
    }
    
    /* Better button styles */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    /* Search box */
    .search-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .search-box:focus {
        outline: none;
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Copy button */
    .copy-button {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        background: #667eea;
        color: white;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        border: none;
        font-weight: 500;
    }
    
    .copy-button:hover {
        background: #5a6fd8;
        transform: scale(1.05);
    }
    
    /* Preset badge */
    .preset-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        background: #ffc107;
        color: #212529;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    /* Accessibility - High contrast mode support */
    @media (prefers-contrast: high) {
        .mentee-card, .match-card {
            border-width: 3px;
        }
    }
    
    /* Accessibility - Reduced motion */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* Better focus indicators for accessibility */
    button:focus, input:focus, textarea:focus {
        outline: 3px solid #667eea;
        outline-offset: 2px;
    }
    
    /* Stats card */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    /* Help panel */
    .help-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
    
    /* Session info */
    .session-info {
        background: #f8f9fa;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #6c757d;
        margin: 0.5rem 0;
    }
</style>
"""