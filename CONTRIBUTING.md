# Contributing to AI Fairness and Explainability Toolkit (AFET)

Thank you for your interest in contributing to AFET! This document provides guidelines and instructions for contributing to the project.

## 🌟 Ways to Contribute

There are many ways to contribute to AFET, regardless of your background or expertise:

### 1. Code Contributions

- **Implement new metrics**: Add new fairness or explainability metrics
- **Improve existing implementations**: Optimize code, fix bugs, or enhance functionality
- **Add framework support**: Extend support to additional ML frameworks
- **Develop visualization components**: Create new ways to visualize fairness and explainability

### 2. Documentation

- **Improve existing documentation**: Clarify explanations, fix typos, or add examples
- **Create tutorials**: Develop step-by-step guides for using AFET
- **Write use case examples**: Share real-world applications of fairness and explainability

### 3. Testing

- **Write unit tests**: Ensure individual components work as expected
- **Develop integration tests**: Verify end-to-end workflows
- **Test on different environments**: Ensure compatibility across platforms

### 4. Design

- **Improve UI/UX**: Enhance the user experience of the dashboard
- **Create visualizations**: Design effective ways to communicate complex metrics
- **Develop branding**: Help with logos, color schemes, and visual identity

### 5. Research

- **Propose new metrics**: Suggest novel approaches to measuring fairness or explainability
- **Benchmark existing methods**: Compare different approaches and their effectiveness
- **Literature reviews**: Summarize relevant academic research

### 6. Community

- **Answer questions**: Help others use AFET effectively
- **Review pull requests**: Provide feedback on contributions
- **Organize events**: Host workshops or hackathons around AFET

## 🚀 Getting Started

### Setting Up Your Development Environment

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/TaimoorKhan10/afet.git
   cd afet
   ```
3. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```
5. **Run tests to verify your setup**:
   ```bash
   pytest
   ```

### Development Workflow

1. **Create a new branch for your feature or fix**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes**
3. **Run tests to ensure your changes don't break existing functionality**:
   ```bash
   pytest
   ```
4. **Commit your changes with a descriptive message**:
   ```bash
   git commit -m "Add new fairness metric: equal opportunity difference"
   ```
5. **Push your changes to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a pull request** from your fork to the main repository

## 📝 Pull Request Guidelines

- **One feature or fix per pull request**: Keep your changes focused
- **Include tests**: Add tests for any new functionality
- **Update documentation**: Ensure documentation reflects your changes
- **Follow code style**: Adhere to the project's coding conventions
- **Write a descriptive title and description**: Explain what your changes do and why

## 🧪 Testing Guidelines

- **Write unit tests** for individual components
- **Write integration tests** for end-to-end workflows
- **Test edge cases** and error conditions
- **Ensure tests are deterministic** and don't depend on external resources

## 📚 Documentation Guidelines

- **Use clear, concise language**
- **Include examples** to illustrate concepts
- **Keep API documentation up-to-date**
- **Follow the Google style for docstrings**

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## 🙋 Getting Help

If you have questions or need help, you can:

- **Open an issue** with your question
- **Join our community chat** (link to be added)
- **Email the maintainers** (taimoorkhaniajaznabi2@gmail.com)

## 🎉 Recognition

We value all contributions and will recognize contributors in our documentation and release notes. Significant contributors may be invited to join as maintainers.

---

Thank you for contributing to AFET! Your efforts help make AI more fair, explainable, and ethical for everyone.