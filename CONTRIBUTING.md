# Contributing to CBAM-STN-TPS-YOLO

Thank you for your interest in contributing to the CBAM-STN-TPS-YOLO agricultural object detection project. This document provides guidelines for contributing to the codebase.

## Project Overview

This repository contains the implementation and analysis framework for CBAM-STN-TPS-YOLO architecture designed for agricultural object detection. The project includes dataset analysis, model implementation, statistical validation, and cross-dataset evaluation tools.

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab
- Git
- Required packages (see requirements.txt)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/SatvikPraveen/CBAM-STN-TPS-YOLO.git
cd CBAM-STN-TPS-YOLO
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Contribute

### Types of Contributions Welcome

1. **Model Implementation**: Help complete the CBAM-STN-TPS-YOLO architecture
2. **Dataset Integration**: Add support for new agricultural datasets
3. **Performance Optimization**: Improve inference speed or memory usage
4. **Documentation**: Enhance code documentation and tutorials
5. **Bug Fixes**: Fix issues in existing code
6. **Testing**: Add unit tests and validation scripts
7. **Analysis Tools**: Contribute new evaluation metrics or visualization tools

### Development Workflow

1. **Fork the repository** on GitHub
2. **Create a feature branch** from main:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding standards below
4. **Test your changes** thoroughly
5. **Commit your changes** with descriptive messages:
   ```bash
   git commit -m "Add: Brief description of your changes"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Submit a Pull Request** with detailed description

### Coding Standards

#### Python Code Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Include docstrings for all functions and classes
- Maximum line length: 88 characters (Black formatter standard)

#### Jupyter Notebooks
- Clear markdown headers and explanations
- Remove output cells before committing (use `nbstripout`)
- Include cell documentation for complex analyses
- Use consistent plotting styles

#### Documentation
- Use clear, concise language
- Include code examples where appropriate
- Update README.md if adding new features
- Document any new dependencies

### Commit Message Guidelines

Use the following format:
```
Type: Brief description (50 chars max)

Detailed explanation if needed (wrap at 72 chars)
```

Types:
- `Add`: New features or files
- `Fix`: Bug fixes
- `Update`: Modifications to existing features
- `Remove`: Deleted features or files
- `Docs`: Documentation changes
- `Test`: Adding or modifying tests
- `Refactor`: Code restructuring without functionality changes

### Pull Request Guidelines

When submitting a pull request:

1. **Provide clear title and description**
2. **Reference related issues** (if applicable)
3. **Include test results** or validation evidence
4. **Update documentation** as needed
5. **Ensure all notebooks run without errors**
6. **Add yourself to CONTRIBUTORS.md**

#### PR Template
```
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (specify)

## Testing
- [ ] All existing tests pass
- [ ] New tests added (if applicable)
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

### Reporting Issues

When reporting bugs or requesting features:

1. **Search existing issues** first
2. **Use issue templates** provided
3. **Include system information** (Python version, OS, etc.)
4. **Provide reproducible examples** for bugs
5. **Be specific and descriptive**

### Dataset Contributions

When adding new agricultural datasets:

1. **Follow existing data exploration structure** (see notebooks 01-03)
2. **Include metadata documentation**
3. **Provide data source attribution**
4. **Add evaluation metrics specific to the dataset**
5. **Update cross-dataset analysis accordingly**

### Model Implementation Guidelines

For model-related contributions:

1. **Maintain modular architecture**
2. **Include comprehensive docstrings**
3. **Add unit tests for new components**
4. **Document hyperparameters and training procedures**
5. **Provide benchmark results**

## Code Review Process

1. All contributions require review by project maintainers
2. Reviews focus on code quality, documentation, and compatibility
3. Feedback should be addressed promptly
4. Maintainers may request changes or provide suggestions
5. Once approved, contributions will be merged

## Community Guidelines

- Be respectful and professional in all interactions
- Help newcomers and answer questions constructively
- Focus on the technical merits of contributions
- Acknowledge the work of others appropriately
- Follow academic integrity standards for research-related contributions

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Repository contributors section
- Academic publications (for significant contributions)
- Release notes for major contributions

## Questions and Support

- Create an issue for technical questions
- Use discussions for general questions about the project
- Contact maintainers directly for sensitive matters

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project [license](LICENSE).

---

Thank you for contributing to advancing agricultural object detection research!
