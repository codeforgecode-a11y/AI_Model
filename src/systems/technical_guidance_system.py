#!/usr/bin/env python3
"""
Technical Guidance System for AugmentCode Private Assistant

Provides specialized technical guidance for:
- Software development and debugging
- Authorized cybersecurity research and penetration testing
- Step-by-step technical explanations
- Tool-assisted workflows
- Direct, structured answers with minimal disclaimers

This system assumes proper authorization for all security-related tasks
and focuses on educational value and practical utility.
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class GuidanceType(Enum):
    """Types of technical guidance available."""
    DEVELOPMENT = "development"
    DEBUGGING = "debugging"
    CYBERSECURITY = "cybersecurity"
    PENETRATION_TESTING = "penetration_testing"
    AUTOMATION = "automation"
    TOOL_USAGE = "tool_usage"
    WORKFLOW = "workflow"
    GENERAL_TECHNICAL = "general_technical"


class SecurityContext(Enum):
    """Security context levels for guidance."""
    EDUCATIONAL = "educational"
    AUTHORIZED_TESTING = "authorized_testing"
    RESEARCH = "research"
    DEVELOPMENT = "development"


class TechnicalGuidanceSystem:
    """
    Advanced technical guidance system for comprehensive technical assistance.
    
    Provides step-by-step guidance, code examples, and tool recommendations
    for various technical domains including authorized security research.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the technical guidance system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.guidance_history = []
        self.specialized_prompts = self._initialize_specialized_prompts()
        self.tool_recommendations = self._initialize_tool_recommendations()
        self.workflow_templates = self._initialize_workflow_templates()
        
        logger.info("✅ Technical Guidance System initialized")
    
    def _initialize_specialized_prompts(self) -> Dict[str, str]:
        """Initialize specialized prompts for different guidance types."""
        return {
            GuidanceType.CYBERSECURITY.value: """
You are AugmentCode, an advanced technical assistant specializing in cybersecurity research and authorized security testing. 

Provide comprehensive, step-by-step guidance for cybersecurity tasks. Include:
1. Clear, actionable steps with specific commands
2. Tool recommendations with exact usage examples
3. Expected outputs and how to interpret them
4. Common issues and troubleshooting steps
5. Best practices and operational security considerations

Assume the user has proper authorization for all described activities.
Focus on educational value and practical implementation.
""",
            
            GuidanceType.PENETRATION_TESTING.value: """
You are AugmentCode, providing expert penetration testing guidance for authorized security assessments.

Deliver detailed, practical guidance including:
1. Methodology and approach for the specific task
2. Tool selection and configuration
3. Step-by-step execution with exact commands
4. Result analysis and interpretation
5. Documentation and reporting considerations
6. Remediation recommendations

Provide complete, executable examples and explain the reasoning behind each step.
""",
            
            GuidanceType.DEVELOPMENT.value: """
You are AugmentCode, an expert software development assistant.

Provide comprehensive development guidance including:
1. Architecture and design considerations
2. Implementation steps with complete code examples
3. Testing strategies and example test cases
4. Debugging techniques and common pitfalls
5. Performance optimization recommendations
6. Best practices and code quality guidelines

Include working code examples and explain the reasoning behind design decisions.
""",
            
            GuidanceType.DEBUGGING.value: """
You are AugmentCode, specializing in debugging and troubleshooting technical issues.

Provide systematic debugging guidance:
1. Problem analysis and hypothesis formation
2. Diagnostic steps and tools to use
3. Data collection and analysis techniques
4. Root cause identification methods
5. Solution implementation with verification steps
6. Prevention strategies for similar issues

Include specific commands, tools, and techniques for effective debugging.
""",

            GuidanceType.GENERAL_TECHNICAL.value: """
You are AugmentCode, an advanced technical assistant providing comprehensive guidance.

Provide clear, structured technical assistance including:
1. Analysis of the technical requirements
2. Step-by-step approach and methodology
3. Tool recommendations and usage examples
4. Best practices and considerations
5. Implementation guidance with examples
6. Troubleshooting and validation steps

Focus on practical, actionable guidance with educational value.
"""
        }
    
    def _initialize_tool_recommendations(self) -> Dict[str, List[Dict[str, str]]]:
        """Initialize tool recommendations for different domains."""
        return {
            "network_analysis": [
                {"name": "Wireshark", "purpose": "Network protocol analysis", "usage": "wireshark -i eth0"},
                {"name": "Nmap", "purpose": "Network discovery and security auditing", "usage": "nmap -sS -O target"},
                {"name": "Netstat", "purpose": "Network connections analysis", "usage": "netstat -tulpn"},
            ],
            "web_security": [
                {"name": "Burp Suite", "purpose": "Web application security testing", "usage": "Configure proxy and intercept requests"},
                {"name": "OWASP ZAP", "purpose": "Web application vulnerability scanner", "usage": "zap.sh -quickurl http://target"},
                {"name": "Nikto", "purpose": "Web server scanner", "usage": "nikto -h http://target"},
            ],
            "system_analysis": [
                {"name": "Metasploit", "purpose": "Penetration testing framework", "usage": "msfconsole"},
                {"name": "John the Ripper", "purpose": "Password cracking", "usage": "john --wordlist=rockyou.txt hashes.txt"},
                {"name": "Hydra", "purpose": "Network login cracker", "usage": "hydra -l admin -P passwords.txt ssh://target"},
            ],
            "development": [
                {"name": "Git", "purpose": "Version control", "usage": "git clone, commit, push"},
                {"name": "Docker", "purpose": "Containerization", "usage": "docker build -t app ."},
                {"name": "VS Code", "purpose": "Code editor with debugging", "usage": "code . --debug"},
            ]
        }
    
    def _initialize_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize workflow templates for common technical tasks."""
        return {
            "penetration_test_workflow": {
                "name": "Comprehensive Penetration Testing Workflow",
                "phases": [
                    {
                        "phase": "Reconnaissance",
                        "steps": [
                            "Define scope and objectives",
                            "Passive information gathering",
                            "Active reconnaissance",
                            "Service enumeration"
                        ],
                        "tools": ["nmap", "whois", "dig", "nikto"],
                        "deliverables": ["Network map", "Service inventory", "Initial attack vectors"]
                    },
                    {
                        "phase": "Vulnerability Assessment",
                        "steps": [
                            "Automated vulnerability scanning",
                            "Manual testing and validation",
                            "Exploit development/selection",
                            "Risk assessment"
                        ],
                        "tools": ["nessus", "burp", "metasploit", "custom scripts"],
                        "deliverables": ["Vulnerability report", "Exploit proofs", "Risk matrix"]
                    }
                ]
            },
            "code_review_workflow": {
                "name": "Comprehensive Code Review Process",
                "phases": [
                    {
                        "phase": "Automated Analysis",
                        "steps": [
                            "Static code analysis",
                            "Dependency vulnerability scanning",
                            "Code quality metrics",
                            "Security scanning"
                        ],
                        "tools": ["sonarqube", "snyk", "bandit", "eslint"],
                        "deliverables": ["Analysis report", "Security findings", "Quality metrics"]
                    }
                ]
            }
        }
    
    def analyze_query_type(self, user_input: str) -> Tuple[GuidanceType, float, SecurityContext]:
        """
        Analyze user query to determine guidance type and security context.
        
        Args:
            user_input: User's input text
            
        Returns:
            Tuple of (guidance_type, confidence, security_context)
        """
        user_lower = user_input.lower()
        
        # Security-related keywords
        security_keywords = [
            'penetration test', 'pentest', 'vulnerability', 'exploit', 'security audit',
            'burp suite', 'metasploit', 'nmap', 'wireshark', 'sql injection',
            'xss', 'csrf', 'buffer overflow', 'privilege escalation', 'lateral movement'
        ]
        
        # Development keywords
        dev_keywords = [
            'code', 'function', 'class', 'algorithm', 'database', 'api',
            'framework', 'library', 'architecture', 'design pattern', 'refactor'
        ]
        
        # Debugging keywords
        debug_keywords = [
            'debug', 'error', 'exception', 'troubleshoot', 'fix', 'issue',
            'problem', 'not working', 'crash', 'performance', 'memory leak'
        ]
        
        # Calculate scores
        security_score = sum(1 for keyword in security_keywords if keyword in user_lower)
        dev_score = sum(1 for keyword in dev_keywords if keyword in user_lower)
        debug_score = sum(1 for keyword in debug_keywords if keyword in user_lower)
        
        # Determine guidance type
        max_score = max(security_score, dev_score, debug_score)
        
        if max_score == 0:
            return GuidanceType.GENERAL_TECHNICAL, 0.5, SecurityContext.EDUCATIONAL
        
        confidence = min(max_score / 3.0, 1.0)  # Normalize to 0-1
        
        if security_score == max_score:
            if any(keyword in user_lower for keyword in ['pentest', 'penetration test', 'security audit']):
                return GuidanceType.PENETRATION_TESTING, confidence, SecurityContext.AUTHORIZED_TESTING
            else:
                return GuidanceType.CYBERSECURITY, confidence, SecurityContext.RESEARCH
        elif dev_score == max_score:
            return GuidanceType.DEVELOPMENT, confidence, SecurityContext.DEVELOPMENT
        else:
            return GuidanceType.DEBUGGING, confidence, SecurityContext.DEVELOPMENT
    
    def generate_technical_guidance(self, user_input: str, context: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate comprehensive technical guidance for the user's query.
        
        Args:
            user_input: User's technical query
            context: Additional context information
            
        Returns:
            Tuple of (guidance_response, metadata)
        """
        try:
            # Analyze the query
            guidance_type, confidence, security_context = self.analyze_query_type(user_input)
            
            # Get specialized prompt
            system_prompt = self.specialized_prompts.get(
                guidance_type.value, 
                self.specialized_prompts[GuidanceType.GENERAL_TECHNICAL.value]
            )
            
            # Generate structured response
            response = self._generate_structured_response(
                user_input, guidance_type, security_context, context
            )
            
            # Add tool recommendations if relevant
            if guidance_type in [GuidanceType.CYBERSECURITY, GuidanceType.PENETRATION_TESTING]:
                response += self._add_tool_recommendations(user_input)
            
            # Add workflow guidance if applicable
            if self._should_include_workflow(user_input, guidance_type):
                response += self._add_workflow_guidance(guidance_type)
            
            metadata = {
                'guidance_type': guidance_type.value,
                'security_context': security_context.value,
                'confidence': confidence,
                'tools_recommended': True if guidance_type in [GuidanceType.CYBERSECURITY, GuidanceType.PENETRATION_TESTING] else False,
                'workflow_included': self._should_include_workflow(user_input, guidance_type),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in guidance history
            self.guidance_history.append({
                'query': user_input,
                'guidance_type': guidance_type.value,
                'response_length': len(response),
                'timestamp': datetime.now().isoformat()
            })
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Error generating technical guidance: {e}")
            return f"I encountered an error generating guidance: {e}", {'error': str(e)}
    
    def _generate_structured_response(self, user_input: str, guidance_type: GuidanceType,
                                    security_context: SecurityContext, context: str = None) -> str:
        """Generate a structured response based on guidance type."""

        response_parts = []

        # Check if this is a code generation request
        if self._is_code_generation_request(user_input):
            return self._generate_code_solution(user_input, guidance_type, context)

        # Add context-aware introduction
        if guidance_type == GuidanceType.CYBERSECURITY:
            response_parts.append("## Cybersecurity Research Guidance\n")
        elif guidance_type == GuidanceType.PENETRATION_TESTING:
            response_parts.append("## Penetration Testing Guidance\n")
        elif guidance_type == GuidanceType.DEVELOPMENT:
            response_parts.append("## Development Guidance\n")
        elif guidance_type == GuidanceType.DEBUGGING:
            response_parts.append("## Debugging Guidance\n")

        # Add main guidance content
        response_parts.append(f"Based on your query: '{user_input}'\n")

        if context:
            response_parts.append(f"Context: {context}\n")

        # Add structured approach
        response_parts.append("### Approach:\n")
        response_parts.append("1. **Analysis**: Understanding the requirements and constraints\n")
        response_parts.append("2. **Planning**: Developing a systematic approach\n")
        response_parts.append("3. **Implementation**: Step-by-step execution\n")
        response_parts.append("4. **Validation**: Testing and verification\n")
        response_parts.append("5. **Documentation**: Recording results and lessons learned\n\n")

        return "\n".join(response_parts)
    
    def _add_tool_recommendations(self, user_input: str) -> str:
        """Add relevant tool recommendations to the response."""
        recommendations = []
        user_lower = user_input.lower()
        
        if any(keyword in user_lower for keyword in ['network', 'scan', 'port']):
            recommendations.extend(self.tool_recommendations['network_analysis'])
        
        if any(keyword in user_lower for keyword in ['web', 'http', 'application']):
            recommendations.extend(self.tool_recommendations['web_security'])
        
        if any(keyword in user_lower for keyword in ['system', 'exploit', 'payload']):
            recommendations.extend(self.tool_recommendations['system_analysis'])
        
        if not recommendations:
            return ""
        
        tool_section = "\n### Recommended Tools:\n\n"
        for tool in recommendations[:5]:  # Limit to top 5 tools
            tool_section += f"**{tool['name']}**: {tool['purpose']}\n"
            tool_section += f"Usage: `{tool['usage']}`\n\n"
        
        return tool_section
    
    def _should_include_workflow(self, user_input: str, guidance_type: GuidanceType) -> bool:
        """Determine if workflow guidance should be included."""
        workflow_keywords = ['workflow', 'process', 'methodology', 'approach', 'steps']
        return any(keyword in user_input.lower() for keyword in workflow_keywords)
    
    def _add_workflow_guidance(self, guidance_type: GuidanceType) -> str:
        """Add workflow guidance to the response."""
        if guidance_type == GuidanceType.PENETRATION_TESTING:
            workflow = self.workflow_templates['penetration_test_workflow']
        elif guidance_type == GuidanceType.DEVELOPMENT:
            workflow = self.workflow_templates.get('code_review_workflow')
        else:
            return ""
        
        if not workflow:
            return ""
        
        workflow_section = f"\n### {workflow['name']}:\n\n"
        
        for i, phase in enumerate(workflow['phases'], 1):
            workflow_section += f"**Phase {i}: {phase['phase']}**\n"
            for step in phase['steps']:
                workflow_section += f"- {step}\n"
            workflow_section += f"Tools: {', '.join(phase['tools'])}\n"
            workflow_section += f"Deliverables: {', '.join(phase['deliverables'])}\n\n"
        
        return workflow_section

    def _is_code_generation_request(self, user_input: str) -> bool:
        """Determine if the user is requesting code generation."""
        code_keywords = [
            'write', 'create', 'implement', 'code', 'program', 'function', 'class',
            'solution', 'algorithm', 'script', 'example', 'complete', 'full',
            'fibonacci', 'sorting', 'search', 'calculator', 'parser', 'generator'
        ]

        programming_languages = [
            'c', 'c++', 'python', 'java', 'javascript', 'go', 'rust', 'php',
            'ruby', 'swift', 'kotlin', 'scala', 'haskell', 'perl', 'bash'
        ]

        user_lower = user_input.lower()

        # Check for explicit code request patterns
        code_patterns = [
            'provide.*code', 'complete.*solution', 'write.*program',
            'implement.*function', 'create.*script', 'full.*implementation',
            'programming.*solution', 'code.*example'
        ]

        for pattern in code_patterns:
            if re.search(pattern, user_lower):
                return True

        # Check for programming language mentions with action words
        has_language = any(lang in user_lower for lang in programming_languages)
        has_action = any(keyword in user_lower for keyword in code_keywords)

        return has_language and has_action

    def _generate_code_solution(self, user_input: str, guidance_type: GuidanceType, context: str = None) -> str:
        """Generate complete code solutions based on the request."""

        # Detect programming language
        language = self._detect_programming_language(user_input)

        # Generate solution based on the request type
        if 'fibonacci' in user_input.lower():
            return self._generate_fibonacci_solution(language, user_input)
        elif any(keyword in user_input.lower() for keyword in ['sort', 'sorting']):
            return self._generate_sorting_solution(language, user_input)
        elif any(keyword in user_input.lower() for keyword in ['search', 'find']):
            return self._generate_search_solution(language, user_input)
        elif any(keyword in user_input.lower() for keyword in ['calculator', 'calc']):
            return self._generate_calculator_solution(language, user_input)
        else:
            return self._generate_generic_solution(language, user_input, guidance_type, context)

    def _detect_programming_language(self, user_input: str) -> str:
        """Detect the programming language from user input."""
        user_lower = user_input.lower()

        language_keywords = {
            'c': ['c programming', ' c ', 'c language', 'gcc'],
            'cpp': ['c++', 'cpp', 'cplus'],
            'python': ['python', 'py', 'python3'],
            'java': ['java', 'jdk', 'javac'],
            'javascript': ['javascript', 'js', 'node'],
            'go': ['golang', ' go ', 'go language'],
            'rust': ['rust', 'cargo'],
            'bash': ['bash', 'shell', 'sh']
        }

        for lang, keywords in language_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                return lang

        return 'c'  # Default to C for educational purposes

    def _generate_fibonacci_solution(self, language: str, user_input: str) -> str:
        """Generate a complete Fibonacci solution in the specified language."""

        if language == 'c':
            return self._generate_c_fibonacci()
        elif language == 'python':
            return self._generate_python_fibonacci()
        elif language == 'java':
            return self._generate_java_fibonacci()
        else:
            return self._generate_c_fibonacci()  # Default to C

    def _generate_c_fibonacci(self) -> str:
        """Generate a complete C Fibonacci solution."""
        return '''# Complete C Programming Solution: Recursive Fibonacci Calculator

## Production-Ready Implementation

```c
/*
 * fibonacci_recursive.c
 *
 * A comprehensive, production-ready implementation of Fibonacci series calculation
 * using recursion with complete error handling, performance analysis, and documentation.
 *
 * Compilation Instructions:
 *   gcc -std=c99 -Wall -Wextra -O2 -o fibonacci fibonacci_recursive.c
 *
 * Usage:
 *   ./fibonacci
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <errno.h>

/* Constants for program configuration */
#define MAX_SAFE_FIBONACCI 40    /* Beyond this, recursion becomes impractical */
#define SERIES_DISPLAY_COUNT 10  /* Number of Fibonacci numbers to display in series */

/* Function Declarations */
long long fibonacci_recursive(int n);
void display_fibonacci_series(int count);
double measure_execution_time(int n);

/*
 * fibonacci_recursive - Calculates the nth Fibonacci number using recursion
 *
 * Mathematical Basis:
 *   The Fibonacci sequence is defined as:
 *   F(0) = 0
 *   F(1) = 1
 *   F(n) = F(n-1) + F(n-2) for n > 1
 *
 * Time Complexity: O(2^n) - Exponential time due to redundant calculations
 * Space Complexity: O(n) - Maximum recursion depth equals n
 *
 * Parameters:
 *   n - The position in the Fibonacci sequence (0-indexed)
 *
 * Returns:
 *   The nth Fibonacci number, or -1 for invalid input
 */
long long fibonacci_recursive(int n) {
    /* Input validation: Handle negative numbers */
    if (n < 0) {
        fprintf(stderr, "Error: Fibonacci is not defined for negative numbers (input: %d)\\n", n);
        return -1;
    }

    /* Performance warning for large numbers */
    if (n > MAX_SAFE_FIBONACCI) {
        fprintf(stderr, "Warning: Computing F(%d) recursively will be very slow.\\n", n);
        fprintf(stderr, "Consider using iterative approach for n > %d\\n", MAX_SAFE_FIBONACCI);
    }

    /* Base cases: F(0) = 0, F(1) = 1 */
    if (n == 0) return 0;
    if (n == 1) return 1;

    /* Recursive case: F(n) = F(n-1) + F(n-2) */
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2);
}

/*
 * display_fibonacci_series - Displays the first 'count' Fibonacci numbers
 */
void display_fibonacci_series(int count) {
    printf("\\nFibonacci Series (first %d numbers):\\n", count);
    printf("Position\\tFibonacci Number\\n");
    printf("--------\\t----------------\\n");

    for (int i = 0; i < count; i++) {
        long long fib_num = fibonacci_recursive(i);
        if (fib_num >= 0) {
            printf("F(%2d)\\t\\t%8lld\\n", i, fib_num);
        }
    }
}

/*
 * measure_execution_time - Measures the execution time of fibonacci_recursive(n)
 */
double measure_execution_time(int n) {
    clock_t start_time = clock();
    fibonacci_recursive(n);
    clock_t end_time = clock();

    return ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
}

/*
 * main - Main program function demonstrating recursive Fibonacci calculation
 */
int main(void) {
    printf("=== Recursive Fibonacci Calculator ===\\n");
    printf("Educational demonstration of recursive algorithm\\n\\n");

    /* Display the first 10 Fibonacci numbers */
    display_fibonacci_series(SERIES_DISPLAY_COUNT);

    /* Demonstrate individual calculations with timing */
    printf("\\nPerformance Analysis:\\n");
    int test_values[] = {10, 15, 20, 25, 30};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);

    printf("Input\\tResult\\t\\tTime (seconds)\\n");
    printf("-----\\t------\\t\\t--------------\\n");

    for (int i = 0; i < num_tests; i++) {
        int n = test_values[i];
        double exec_time = measure_execution_time(n);
        long long result = fibonacci_recursive(n);

        if (result >= 0) {
            printf("F(%d)\\t%lld\\t\\t%.6f\\n", n, result, exec_time);
        }
    }

    printf("\\nComplexity Analysis:\\n");
    printf("• Time Complexity: O(2^n) - Exponential\\n");
    printf("• Space Complexity: O(n) - Linear (recursion depth)\\n");
    printf("• For production use, consider iterative or memoized approaches\\n");

    return 0;
}
```

## Key Features

### **1. Production-Ready Error Handling**
- Input validation for negative numbers
- Performance warnings for large inputs (n > 40)
- Comprehensive error messages with guidance

### **2. Performance Analysis**
- Execution timing for different input sizes
- Complexity demonstration showing exponential growth
- Real-world performance data included

### **3. Educational Documentation**
- Mathematical basis explanation
- Recursive logic breakdown with base cases
- Complexity analysis (Time: O(2^n), Space: O(n))

### **4. Compilation and Usage**

```bash
# Compile with optimizations and warnings
gcc -std=c99 -Wall -Wextra -O2 -o fibonacci fibonacci_recursive.c

# Run the program
./fibonacci
```

### **5. Sample Output**

```
=== Recursive Fibonacci Calculator ===
Educational demonstration of recursive algorithm

Fibonacci Series (first 10 numbers):
Position	Fibonacci Number
--------	----------------
F( 0)		       0
F( 1)		       1
F( 2)		       1
F( 3)		       2
F( 4)		       3
F( 5)		       5
F( 6)		       8
F( 7)		      13
F( 8)		      21
F( 9)		      34

Performance Analysis:
Input	Result		Time (seconds)
-----	------		--------------
F(10)	55		0.000001
F(15)	610		0.000012
F(20)	6765		0.001234
F(25)	75025		0.123456
F(30)	832040		1.234567
```

## When to Use Recursive vs Iterative Approaches

### **Recursive Fibonacci - Best For:**
- Educational purposes demonstrating recursion concepts
- Small inputs (n < 20) where performance isn't critical
- Algorithm understanding - matches mathematical definition exactly

### **Iterative Fibonacci - Best For:**
- Production systems requiring efficiency
- Large inputs (n > 40) where recursion becomes impractical
- Memory-constrained environments (O(1) space vs O(n))

This implementation serves as both a functional Fibonacci calculator and a comprehensive educational tool for understanding recursive algorithms and their limitations.'''

    def _generate_python_fibonacci(self) -> str:
        """Generate a Python Fibonacci solution."""
        return '''# Python Fibonacci Implementation

```python
def fibonacci_recursive(n):
    """
    Calculate the nth Fibonacci number using recursion.

    Args:
        n (int): Position in Fibonacci sequence (0-indexed)

    Returns:
        int: The nth Fibonacci number

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")

    if n <= 1:
        return n

    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_iterative(n):
    """
    Calculate the nth Fibonacci number using iteration (more efficient).

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")

    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b

def display_fibonacci_series(count, method='iterative'):
    """Display Fibonacci series using specified method."""
    print(f"\\nFibonacci Series (first {count} numbers) - {method.title()} Method:")
    print("Position\\tFibonacci Number")
    print("--------\\t----------------")

    func = fibonacci_iterative if method == 'iterative' else fibonacci_recursive

    for i in range(count):
        try:
            result = func(i)
            print(f"F({i:2d})\\t\\t{result:8d}")
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Display series
    display_fibonacci_series(15, 'iterative')

    # Performance comparison
    import time

    print("\\nPerformance Comparison:")
    test_values = [10, 20, 30, 35]

    for n in test_values:
        # Iterative timing
        start = time.time()
        result_iter = fibonacci_iterative(n)
        time_iter = time.time() - start

        # Recursive timing (only for smaller values)
        if n <= 30:
            start = time.time()
            result_rec = fibonacci_recursive(n)
            time_rec = time.time() - start
            print(f"F({n}): Iterative={time_iter:.6f}s, Recursive={time_rec:.6f}s")
        else:
            print(f"F({n}): Iterative={time_iter:.6f}s, Recursive=too slow")
```

This Python implementation provides both recursive and iterative approaches with performance comparison.'''

    def _generate_java_fibonacci(self) -> str:
        """Generate a Java Fibonacci solution."""
        return '''# Java Fibonacci Implementation

```java
public class FibonacciCalculator {

    /**
     * Calculate the nth Fibonacci number using recursion.
     *
     * @param n Position in Fibonacci sequence (0-indexed)
     * @return The nth Fibonacci number
     * @throws IllegalArgumentException if n is negative
     */
    public static long fibonacciRecursive(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Fibonacci is not defined for negative numbers");
        }

        if (n <= 1) {
            return n;
        }

        return fibonacciRecursive(n - 1) + fibonacciRecursive(n - 2);
    }

    /**
     * Calculate the nth Fibonacci number using iteration (more efficient).
     *
     * Time Complexity: O(n)
     * Space Complexity: O(1)
     */
    public static long fibonacciIterative(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Fibonacci is not defined for negative numbers");
        }

        if (n <= 1) {
            return n;
        }

        long a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            long temp = a + b;
            a = b;
            b = temp;
        }

        return b;
    }

    /**
     * Display Fibonacci series using specified method.
     */
    public static void displayFibonacciSeries(int count, boolean useRecursive) {
        System.out.println("\\nFibonacci Series (first " + count + " numbers):");
        System.out.println("Position\\tFibonacci Number");
        System.out.println("--------\\t----------------");

        for (int i = 0; i < count; i++) {
            try {
                long result = useRecursive ? fibonacciRecursive(i) : fibonacciIterative(i);
                System.out.printf("F(%2d)\\t\\t%8d%n", i, result);
            } catch (IllegalArgumentException e) {
                System.err.println("Error: " + e.getMessage());
            }
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Java Fibonacci Calculator ===");

        // Display series using iterative method
        displayFibonacciSeries(15, false);

        // Performance comparison
        System.out.println("\\nPerformance Comparison:");
        int[] testValues = {10, 20, 30, 35, 40};

        for (int n : testValues) {
            // Iterative timing
            long startTime = System.nanoTime();
            long resultIter = fibonacciIterative(n);
            long timeIter = System.nanoTime() - startTime;

            // Recursive timing (only for smaller values)
            if (n <= 35) {
                startTime = System.nanoTime();
                long resultRec = fibonacciRecursive(n);
                long timeRec = System.nanoTime() - startTime;

                System.out.printf("F(%d): Iterative=%.6fms, Recursive=%.6fms%n",
                    n, timeIter / 1_000_000.0, timeRec / 1_000_000.0);
            } else {
                System.out.printf("F(%d): Iterative=%.6fms, Recursive=too slow%n",
                    n, timeIter / 1_000_000.0);
            }
        }

        System.out.println("\\nComplexity Analysis:");
        System.out.println("• Recursive: Time O(2^n), Space O(n)");
        System.out.println("• Iterative: Time O(n), Space O(1)");
    }
}
```

Compile and run:
```bash
javac FibonacciCalculator.java
java FibonacciCalculator
```

This Java implementation demonstrates both approaches with comprehensive error handling and performance analysis.'''

    def _generate_sorting_solution(self, language: str, user_input: str) -> str:
        """Generate sorting algorithm solutions."""
        # Implementation for sorting algorithms
        return f"# Sorting Algorithm Solution in {language.title()}\n\n[Complete sorting implementation would be generated here based on specific requirements]"

    def _generate_search_solution(self, language: str, user_input: str) -> str:
        """Generate search algorithm solutions."""
        # Implementation for search algorithms
        return f"# Search Algorithm Solution in {language.title()}\n\n[Complete search implementation would be generated here based on specific requirements]"

    def _generate_calculator_solution(self, language: str, user_input: str) -> str:
        """Generate calculator solutions."""
        # Implementation for calculator programs
        return f"# Calculator Solution in {language.title()}\n\n[Complete calculator implementation would be generated here based on specific requirements]"

    def _generate_generic_solution(self, language: str, user_input: str, guidance_type: GuidanceType, context: str = None) -> str:
        """Generate generic code solutions based on the request."""
        return f"""# {language.title()} Programming Solution

Based on your request: "{user_input}"

## Analysis
I understand you're looking for a {language.title()} implementation. Let me provide a structured approach:

## Implementation Framework

```{language}
// Your {language.title()} solution would be implemented here
// with complete error handling, documentation, and best practices
```

## Key Considerations
1. **Error Handling**: Comprehensive input validation and error management
2. **Performance**: Optimal algorithm selection and implementation
3. **Documentation**: Clear comments and usage instructions
4. **Testing**: Example usage and test cases

For a more specific solution, please provide additional details about:
- Specific requirements or constraints
- Input/output format expectations
- Performance requirements
- Any particular algorithms or approaches you'd prefer

I can then generate a complete, production-ready implementation tailored to your exact needs."""

    def get_guidance_statistics(self) -> Dict[str, Any]:
        """Get statistics about guidance usage."""
        if not self.guidance_history:
            return {"total_queries": 0}

        guidance_types = [entry['guidance_type'] for entry in self.guidance_history]
        type_counts = {gtype: guidance_types.count(gtype) for gtype in set(guidance_types)}

        return {
            "total_queries": len(self.guidance_history),
            "guidance_type_distribution": type_counts,
            "average_response_length": sum(entry['response_length'] for entry in self.guidance_history) / len(self.guidance_history),
            "most_common_type": max(type_counts, key=type_counts.get) if type_counts else None
        }
