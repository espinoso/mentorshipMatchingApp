# 🤝 Mentorship Matching System

An AI-powered web application that intelligently matches mentees with mentors based on expertise, skills, career goals, and professional experience. Built with Streamlit and powered by OpenAI's GPT models.

## 🌟 Features

- **Intelligent Matching**: Uses AI to analyze compatibility between mentees and mentors
- **Flexible Data Input**: Supports Excel file uploads for mentees, mentors, and training data
- **Customizable Prompts**: Modify AI matching criteria through an intuitive interface
- **Conflict Detection**: Automatically identifies mentor assignment conflicts
- **Interactive Results**: Browse matches with detailed reasoning and match percentages
- **Export Functionality**: Download comprehensive Excel reports
- **Sample Data**: Built-in sample data for testing and demonstration

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llmWebInterface
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📋 Data Requirements

### Mentee File Structure
Your mentee Excel file should contain the following columns:
- `Code Mentee` - Unique identifier for the mentee
- `Mentee Field of expertise` - Primary field of study/work
- `Mentee Specialization` - Specific area within the field
- `Mentee Specific hard skills that you have` - Technical skills and competencies
- `Mentee Areas where guidance is needed` - What the mentee wants to learn
- `Mentee Career goals for the next 2 years` - Short-term career objectives

### Mentor File Structure
Your mentor Excel file should contain the following columns:
- `Code mentor` - Unique identifier for the mentor
- `Field of expertise` - Primary field of expertise
- `Mentor Specialization` - Specific areas of specialization
- `Mentor Specific Hard Skills and Professional Competencies has Mastered` - Skills and competencies
- `Mentor Years of Professional Experience in his her Field` - Years of experience

### Training Files (Optional)
Upload historical mentorship data to improve matching accuracy. These files should contain both mentee and mentor information from successful past matches.

## 🎯 How to Use

### 1. Upload Data
- **Training Files**: Upload historical mentorship data (optional but recommended)
- **Mentees File**: Upload your mentee data in Excel format
- **Mentors File**: Upload your mentor data in Excel format

### 2. Configure AI
- Enter your OpenAI API key
- Optionally customize the matching prompt in the "Customize Prompt" tab

### 3. Generate Matches
- Click "Generate Matches" to run the AI matching algorithm
- Review results and check for any mentor assignment conflicts

### 4. Export Results
- Download comprehensive Excel reports with all matching results
- Reports include match percentages, reasoning, and original data

## 🔧 Customization

### Modifying Matching Criteria
You can customize the AI's matching behavior by editing the prompt in the "Customize Prompt" tab. The default prompt considers:

- Field of expertise alignment
- Specialization overlap
- Hard skills compatibility
- Career goals alignment
- Guidance needs matching
- Professional experience appropriateness

### Sample Data
Use the "Use Sample Data" button to test the system with pre-loaded sample data that demonstrates the expected format and functionality.

## 📊 Output Format

The system generates matches in the following format:
```json
{
  "mentee_id": "Mentee_A1",
  "matches": [
    {
      "rank": 1,
      "mentor_id": "Mentor_X1",
      "match_percentage": 85,
      "match_quality": "Excellent",
      "reasoning": "Detailed explanation of compatibility..."
    }
  ]
}
```

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **OpenAI**: AI model integration
- **OpenPyXL**: Excel file handling

### Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Python with OpenAI API integration
- **Data Processing**: Pandas for data cleaning and validation
- **AI Engine**: OpenAI GPT-4o-mini for intelligent matching

## 🔒 Security & Privacy

- API keys are handled securely and not stored
- Data processing happens locally
- No data is permanently stored on external servers
- All uploaded files remain in your local environment

## 🐛 Troubleshooting

### Common Issues

1. **Missing Required Columns**
   - Ensure your Excel files contain all required columns
   - Check column names match exactly (case-sensitive)

2. **API Key Issues**
   - Verify your OpenAI API key is valid and has sufficient credits
   - Ensure you have access to GPT-4o-mini model

3. **File Upload Errors**
   - Check that files are in Excel (.xlsx) format
   - Ensure files are not corrupted or password-protected

4. **Memory Issues with Large Files**
   - Consider splitting large datasets into smaller files
   - The system works best with datasets under 1000 records

## 📈 Performance Tips

- **Training Data**: Include relevant historical data to improve matching accuracy
- **Data Quality**: Clean and consistent data produces better matches
- **Prompt Tuning**: Experiment with different prompts for your specific use case
- **Batch Processing**: Process mentees in smaller batches for large datasets

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section above
- Review the sample data format for reference

---

**Built with ❤️ for better mentorship matching**
