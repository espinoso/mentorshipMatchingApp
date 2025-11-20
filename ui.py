CSS_STYLES = """
<style>
    /* Import IMFAHE brand font - Open Sans */
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600&display=swap');
    
    /* Global styles - IMFAHE brand typography */
    * {
        font-family: 'Open Sans', 'Noto Sans', Helvetica, Arial, sans-serif;
        letter-spacing: -0.3px;
    }
    
    /* IMFAHE Brand Colors */
    /* Primary: #E6122C (Red), Accent: #F39220 (Orange), Text: #3A3A3A, BG: #FFFFFF */
    
    /* Professional main header - White box with red text and border (subtle) */
    .main-header {
        text-align: center;
        padding: 0.75rem 0;
        background: #FFFFFF;
        color: #E6122C;
        border: 2px solid #E6122C;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
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
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.3px;
        color: #E6122C;
    }
    
    .main-header p {
        font-weight: 400;
        font-size: 0.85rem;
        opacity: 0.9;
        margin: 0;
        color: #3A3A3A;
    }
    
    /* Sticky header */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: white;
        padding: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: -1rem -1rem 1rem -1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Professional mentee card - IMFAHE red accent */
    .mentee-card {
        background: #FFFFFF;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #E6122C;
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
        background: #E6122C;
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: 0;
    }
    
    .mentee-card:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 24px rgba(230, 18, 44, 0.15);
        border-left-width: 6px;
    }
    
    .mentee-card.selected {
        background: #E6122C !important;
        color: white !important;
        border-left: 6px solid #c01025 !important;
        box-shadow: 0 8px 32px rgba(230, 18, 44, 0.3) !important;
        transform: translateX(12px);
    }
    
    .mentee-card * {
        position: relative;
        z-index: 1;
    }
    
    /* Professional match card - clean white */
    .match-card {
        background: #FFFFFF !important;
        padding: 1.8rem;
        border-radius: 16px;
        border: 2px solid #E2E2E2;
        margin: 1.2rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        color: #3A3A3A !important;
        transition: all 0.3s ease;
        animation: fadeIn 0.4s ease-out;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    .match-card:hover {
        box-shadow: 0 8px 28px rgba(0,0,0,0.1);
        transform: translateY(-2px);
        border-color: #E6122C;
    }
    
    .match-card h4 {
        color: #3A3A3A !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.3px !important;
    }
    
    .match-card p {
        color: #3A3A3A !important;
        margin-bottom: 0.75rem !important;
        line-height: 1.7 !important;
        font-size: 0.95rem !important;
    }
    
    .match-card strong {
        color: #3A3A3A !important;
        font-weight: 600 !important;
    }
    
    /* Professional percentage badge - no pulse animation */
    .percentage-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 24px;
        font-weight: 600;
        color: white;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    
    .excellent { 
        background: #10b981;
    }
    .strong { 
        background: #F39220;
    }
    .good { 
        background: #fcb900;
        color: #3A3A3A;
    }
    .fair { 
        background: #fcb900;
        color: #3A3A3A;
    }
    
    /* Professional alert boxes - no shake/bounce animations */
    .conflict-warning {
        background: #fff5f5;
        color: #721c24;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #E6122C;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(230, 18, 44, 0.1);
    }
    
    .cost-estimate {
        background: #f0fdf4;
        color: #155724;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    
    .token-warning {
        background: #fffbeb;
        color: #856404;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #F39220;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(243, 146, 32, 0.1);
    }
    
    /* Professional wizard steps - IMFAHE colors */
    .wizard-step {
        display: flex;
        align-items: center;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: #F2F2F2;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .wizard-step.active {
        background: #E6122C;
        color: white;
        border-color: #c01025;
        box-shadow: 0 4px 12px rgba(230, 18, 44, 0.2);
    }
    
    .wizard-step.completed {
        background: #f0fdf4;
        border-color: #10b981;
    }
    
    .wizard-step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #E6122C;
        color: white;
        font-weight: 600;
        margin-right: 0.8rem;
        font-size: 0.9rem;
    }
    
    .wizard-step.completed .wizard-step-number {
        background: #10b981;
    }
    
    .wizard-step.active .wizard-step-number {
        background: white;
        color: #E6122C;
    }
    
    /* Professional info card */
    .info-card {
        background: #F5F5F7;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0693e3;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Professional success card */
    .success-card {
        background: #f0fdf4;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.1);
        animation: fadeInUp 0.5s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Loading skeleton */
    .skeleton {
        background: linear-gradient(90deg, #F2F2F2 25%, #E2E2E2 50%, #F2F2F2 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: 8px;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Professional tooltip - IMFAHE red */
    .tooltip-icon {
        display: inline-block;
        width: 18px;
        height: 18px;
        background: #E6122C;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 18px;
        font-size: 12px;
        font-weight: 600;
        cursor: help;
        margin-left: 0.3rem;
    }
    
    /* Professional button styles - IMFAHE colors */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    
    /* Search box */
    .search-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 2px solid #E2E2E2;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .search-box:focus {
        outline: none;
        border-color: #E6122C;
        box-shadow: 0 0 0 3px rgba(230, 18, 44, 0.1);
    }
    
    /* Professional copy button - IMFAHE red */
    .copy-button {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        background: #E6122C;
        color: white;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        border: none;
        font-weight: 500;
    }
    
    .copy-button:hover {
        background: #c01025;
        transform: scale(1.02);
    }
    
    /* Professional preset badge - IMFAHE orange */
    .preset-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        background: #F39220;
        color: white;
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
    
    /* Professional focus indicators - IMFAHE red */
    button:focus, input:focus, textarea:focus {
        outline: 3px solid #E6122C;
        outline-offset: 2px;
    }
    
    /* Professional stats card - IMFAHE red accent */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #E6122C;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    /* Professional help panel */
    .help-panel {
        background: #F5F5F7;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #E2E2E2;
        margin: 1rem 0;
    }
    
    /* Professional session info */
    .session-info {
        background: #F2F2F2;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #3A3A3A;
        margin: 0.5rem 0;
    }
    
    /* Ensure all subheaders are smaller than main header (1.5rem) */
    h2, h3, h4, h5, h6 {
        font-size: 1.3rem !important;
        font-weight: 600;
    }
    
    /* Streamlit subheader override */
    .element-container h2, .element-container h3 {
        font-size: 1.3rem !important;
    }
    
    /* Streamlit metric labels and values - ensure smaller than main header (1.5rem) */
    [data-testid="stMetricLabel"], 
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] > div,
    .stMetric label,
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #3A3A3A !important;
    }
    
    [data-testid="stMetricValue"], 
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] > div > div,
    .stMetric [data-testid="stMetricValue"],
    .stMetric [data-testid="stMetricValue"] > div {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #3A3A3A !important;
    }
    
    /* Streamlit metric container - ensure all text is smaller than main header */
    [data-testid="stMetricContainer"] {
        font-size: 1.2rem !important;
    }
    
    /* Override any large font sizes in metric containers */
    [data-testid="stMetricContainer"] p,
    [data-testid="stMetricContainer"] div,
    [data-testid="stMetricContainer"] span {
        font-size: 1.2rem !important;
    }
    
    /* Ensure metric delta (change indicator) is also smaller */
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }
</style>
"""
