"""
SANY Error Code Reverse Mapping Module

This module provides reverse mapping from SANY error messages to internal error codes.
Since SANY (TLA+ Semantic Analyzer) doesn't output structured error codes like TLC,
we use pattern matching on error messages to infer the original error codes.

Based on TLA+ source code from:
- tla2sany/semantic/ErrorCode.java (error code definitions)
- tla2sany/semantic/Generator.java (error message generation)
"""

import re
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum


@dataclass
class SANYErrorMatch:
    """Result of SANY error code matching"""
    error_code: int
    error_name: str
    confidence: float  # 0.0 to 1.0
    matched_pattern: str
    description: str


class SANYErrorCategory(Enum):
    """SANY error categories based on error code ranges"""
    PARSER_INTERNAL = "parser_internal"      # 4000-4002
    INTERNAL_ERROR = "internal_error"        # 4003-4005  
    BASIC_SEMANTIC = "basic_semantic"        # 4200-4206
    MODULE_IMPORT = "module_import"          # 4220-4224
    INSTANCE_SUBSTITUTION = "instance_substitution"  # 4240-4247
    FUNCTION_RECORD = "function_record"      # 4260-4262
    HIGHER_ORDER = "higher_order"           # 4270-4275
    RECURSIVE = "recursive"                 # 4290-4294
    TEMPORAL = "temporal"                   # 4310-4315
    LABEL = "label"                         # 4330-4337
    PROOF = "proof"                         # 4350-4362
    WARNING = "warning"                     # 4800+
    SYNTAX_PARSE = "syntax_parse"           # 9000+ (custom for common parse errors)


class SANYErrorCodeReverse:
    """
    Reverse engineer SANY error codes from error messages.
    
    This class contains pattern matching rules to identify SANY internal
    error codes based on the error message text, since SANY doesn't output
    structured error codes like TLC does.
    """
    
    def __init__(self):
        self._error_patterns = self._build_error_patterns()
    
    def _build_error_patterns(self) -> Dict[int, Tuple[str, str, str]]:
        """
        Build pattern matching rules for SANY error codes.
        
        Returns:
            Dict mapping error_code -> (pattern, error_name, description)
        """
        patterns = {
            # SANY Basic Semantic Errors (4200-4206)
            4200: (
                r"Unknown operator:\s*[`']([^`']+)[`']\.|DEF\s+clause\s+entry\s+should\s+describe\s+a\s+defined\s+operator",
                "SYMBOL_UNDEFINED", 
                "Symbol or operator not defined"
            ),
            4201: (
                r"Multiply-defined symbol\s+[`']([^`']+)[`']",
                "SYMBOL_REDEFINED",
                "Symbol redefined in same scope"
            ),
            4202: (
                r"Built-in\s+operator\s+[`']?([^`'\s]+)[`']?\s+cannot\s+be\s+redefined",
                "BUILT_IN_SYMBOL_REDEFINED",
                "Built-in TLA+ operator redefined"
            ),
            4203: (
                r"Operator\s+name\s+[`']?(\w+)[`']?\s+is\s+incomplete",
                "OPERATOR_NAME_INCOMPLETE", 
                "Operator name syntax incomplete"
            ),
            4204: (
                r"(?:The\s+operator\s+[`']([^`']+)[`']\s+requires\s+(\d+)\s+arguments?|Wrong\s+number\s+of\s+arguments\s*\((\d+)\)\s*given\s+to\s+operator\s+[`']([^`']+)[`']|Operator\s+used\s+with\s+the\s+wrong\s+number\s+of\s+arguments)",
                "OPERATOR_GIVEN_INCORRECT_NUMBER_OF_ARGUMENTS",
                "Operator called with wrong number of arguments"
            ),
            4205: (
                r"Level\s+constraint\s+(?:of\s+operator\s+)?[`']?(\w+)[`']?\s+(?:is\s+)?exceeded",
                "OPERATOR_LEVEL_CONSTRAINTS_EXCEEDED",
                "TLA+ level constraint violated"
            ),
            4206: (
                r"Assumption\s+(?:[`']?(\w+)[`']?\s+)?is\s+not\s+constant",
                "ASSUMPTION_IS_NOT_CONSTANT",
                "Assumption must be constant-level"
            ),
            
            # SANY Module Import Errors (4220-4224)
            4220: (
                r"Cannot\s+find\s+source\s+file\s+for\s+module\s+[`']([^`']+)[`']\s+in\s+director(?:y|ies)",
                "MODULE_FILE_CANNOT_BE_FOUND",
                "Module file not found"
            ),
            4221: (
                r"File\s+name\s+[`']([^`']+)[`']\s+does\s+not\s+match\s+the\s+name\s+[`']([^`']+)[`']\s+of\s+the\s+top\s+level\s+module\s+it\s+contains",
                "MODULE_NAME_DIFFERENT_FROM_FILE_NAME",
                "Module name doesn't match filename"
            ),
            4222: (
                r"(?:Module\s+)?dependencies\s+are\s+circular",
                "MODULE_DEPENDENCIES_ARE_CIRCULAR",
                "Circular module dependencies detected"
            ),
            4223: (
                r"Module\s+[`']?(\w+)[`']?\s+(?:is\s+)?(?:already\s+)?(?:re)?defined",
                "MODULE_REDEFINED",
                "Module defined multiple times"
            ),
            4224: (
                r"Extended\s+modules?\s+symbol\s+unification\s+conflict",
                "EXTENDED_MODULES_SYMBOL_UNIFICATION_CONFLICT",
                "Symbol conflict in extended modules"
            ),
            
            # SANY Function and Record Errors (4260-4262)
            4260: (
                r"Function\s+[`']?(\w+)[`']?\s+is\s+defined\s+with\s+(\d+)\s+parameters?,\s+but\s+is\s+applied\s+to\s+(\d+)\s+arguments?",
                "FUNCTION_GIVEN_INCORRECT_NUMBER_OF_ARGUMENTS",
                "Function called with wrong number of arguments"
            ),
            4261: (
                r"Function\s+[`']?(\w+)[`']?\s+(?:is\s+)?used\s+(?:with\s+)?EXCEPT\s+(?:at\s+)?(?:a\s+)?(?:point\s+)?(?:where\s+it\s+is\s+)?undefined",
                "FUNCTION_EXCEPT_AT_USED_WHERE_UNDEFINED",
                "EXCEPT used where function undefined"
            ),
            4262: (
                r"Record\s+constructor\s+field\s+[`']?(\w+)[`']?\s+(?:is\s+)?(?:re)?defined\s+(?:multiple\s+times|again)",
                "RECORD_CONSTRUCTOR_FIELD_REDEFINITION", 
                "Record field defined multiple times"
            ),
            
            # SANY Instance Substitution Errors (4240-4247)
            4240: (
                r"(?:Instance\s+)?(?:substitution\s+)?(?:is\s+)?missing\s+symbol\s+[`']?(\w+)[`']?",
                "INSTANCE_SUBSTITUTION_MISSING_SYMBOL",
                "Required substitution symbol missing"
            ),
            4241: (
                r"(?:Instance\s+)?(?:substitution\s+)?symbol\s+[`']?(\w+)[`']?\s+(?:is\s+)?(?:re)?defined\s+multiple\s+times",
                "INSTANCE_SUBSTITUTION_SYMBOL_REDEFINED_MULTIPLE_TIMES",
                "Substitution symbol redefined multiple times"
            ),
            4242: (
                r"(?:Instance\s+)?(?:substitution\s+)?illegal\s+symbol\s+redefinition",
                "INSTANCE_SUBSTITUTION_ILLEGAL_SYMBOL_REDEFINITION",
                "Illegal symbol redefinition in substitution"
            ),
            4243: (
                r"(?:Instance\s+)?(?:substitution\s+)?operator\s+(?:or\s+constant\s+)?(?:has\s+)?incorrect\s+arity",
                "INSTANCE_SUBSTITUTION_OPERATOR_CONSTANT_INCORRECT_ARITY",
                "Operator/constant arity mismatch in substitution"
            ),
            4244: (
                r"(?:Instance\s+)?(?:substitution\s+)?non-Leibniz\s+operator",
                "INSTANCE_SUBSTITUTION_NON_LEIBNIZ_OPERATOR",
                "Non-Leibniz operator in substitution"
            ),
            4245: (
                r"(?:Instance\s+)?(?:substitution\s+)?level\s+constraints?\s+exceeded",
                "INSTANCE_SUBSTITUTION_LEVEL_CONSTRAINTS_EXCEEDED",
                "Level constraints exceeded in substitution"
            ),
            4246: (
                r"(?:Instance\s+)?(?:substitution\s+)?level\s+constraint\s+(?:is\s+)?not\s+met",
                "INSTANCE_SUBSTITUTION_LEVEL_CONSTRAINT_NOT_MET",
                "Level constraint not met in substitution"
            ),
            4247: (
                r"(?:Instance\s+)?(?:substitution\s+)?coparameter\s+level\s+constraints?\s+exceeded",
                "INSTANCE_SUBSTITUTION_COPARAMETER_LEVEL_CONSTRAINTS_EXCEEDED",
                "Coparameter level constraints exceeded"
            ),
            
            # SANY Higher-Order Operator Errors (4270-4275)
            4270: (
                r"Higher-order\s+operator\s+(?:is\s+)?required\s+but\s+expression\s+(?:is\s+)?given",
                "HIGHER_ORDER_OPERATOR_REQUIRED_BUT_EXPRESSION_GIVEN",
                "Expression given where higher-order operator required"
            ),
            4271: (
                r"(?:Expected\s+arity\s+(\d+)\s+but\s+found\s+operator\s+of\s+arity\s+(\d+)|Argument\s+number\s+(\d+)\s+to\s+operator\s+[`']([^`']+)[`']\s+should\s+be\s+a\s+(\d+)-parameter\s+operator)",
                "HIGHER_ORDER_OPERATOR_ARGUMENT_HAS_INCORRECT_ARITY",
                "Higher-order operator argument has incorrect arity"
            ),
            4272: (
                r"Higher-order\s+operator\s+parameter\s+level\s+constraint\s+(?:is\s+)?not\s+met",
                "HIGHER_ORDER_OPERATOR_PARAMETER_LEVEL_CONSTRAINT_NOT_MET",
                "Parameter level constraint not met"
            ),
            4273: (
                r"Higher-order\s+operator\s+coparameter\s+level\s+constraints?\s+exceeded",
                "HIGHER_ORDER_OPERATOR_COPARAMETER_LEVEL_CONSTRAINTS_EXCEEDED",
                "Coparameter level constraints exceeded"
            ),
            4274: (
                r"Selector\s+with\s+(\d+)\s+arguments?\s+used\s+for\s+LAMBDA\s+expression\s+taking\s+(\d+)\s+arguments?",
                "LAMBDA_OPERATOR_ARGUMENT_HAS_INCORRECT_ARITY",
                "LAMBDA operator argument has incorrect arity"
            ),
            4275: (
                r"Lambda\s+(?:operator\s+)?given\s+where\s+expression\s+(?:is\s+)?required",
                "LAMBDA_GIVEN_WHERE_EXPRESSION_REQUIRED",
                "Lambda given where expression required"
            ),
            
            # SANY Recursive Operator Errors (4290-4294)
            4290: (
                r"Recursive\s+operator\s+[`']?(\w+)[`']?\s+(?:has\s+)?primed\s+parameter",
                "RECURSIVE_OPERATOR_PRIMES_PARAMETER",
                "Recursive operator has primed parameter"
            ),
            4291: (
                r"Recursive\s+operator\s+[`']?(\w+)[`']?\s+(?:is\s+)?declared\s+but\s+not\s+defined",
                "RECURSIVE_OPERATOR_DECLARED_BUT_NOT_DEFINED",
                "Recursive operator declared but not defined"
            ),
            4292: (
                r"Recursive\s+operator\s+(?:declaration\s+(?:and\s+)?definition\s+)?arity\s+mismatch",
                "RECURSIVE_OPERATOR_DECLARATION_DEFINITION_ARITY_MISMATCH",
                "Recursive operator declaration/definition arity mismatch"
            ),
            4293: (
                r"Recursive\s+operator\s+[`']?(\w+)[`']?\s+(?:is\s+)?defined\s+in\s+wrong\s+LET-IN\s+level",
                "RECURSIVE_OPERATOR_DEFINED_IN_WRONG_LET_IN_LEVEL",
                "Recursive operator defined at wrong LET-IN level"
            ),
            4294: (
                r"Recursive\s+section\s+contains\s+illegal\s+definition",
                "RECURSIVE_SECTION_CONTAINS_ILLEGAL_DEFINITION",
                "Illegal definition in recursive section"
            ),
            
            # SANY Temporal Operator Errors (4310-4315) - Updated to match actual TLA+ source
            4310: (
                r"\[\]\s+followed\s+by\s+action\s+not\s+of\s+form\s+\[A\]_v",
                "ALWAYS_PROPERTY_SENSITIVE_TO_STUTTERING",
                "[] followed by action not of form [A]_v"
            ),
            4311: (
                r"<>\s+followed\s+by\s+action\s+not\s+of\s+form\s+<<A>>_v",
                "EVENTUALLY_PROPERTY_SENSITIVE_TO_STUTTERING",
                "<> followed by action not of form <<A>>_v"
            ),
            4312: (
                r"(?:Binary\s+temporal\s+operator\s+with\s+action\s+level\s+parameter|Action\s+used\s+where\s+only\s+temporal\s+formula\s+or\s+state\s+formula\s+is\s+allowed)",
                "BINARY_TEMPORAL_OPERATOR_WITH_ACTION_LEVEL_PARAMETER",
                "Action used where temporal/state formula required"
            ),
            4313: (
                r"Logical\s+operator\s+with\s+mixed\s+action\s+(?:and\s+)?temporal\s+parameters",
                "LOGICAL_OPERATOR_WITH_MIXED_ACTION_TEMPORAL_PARAMETERS",
                "Logical operator with mixed action/temporal parameters"
            ),
            4314: (
                r"Quantified\s+temporal\s+formula\s+with\s+action\s+level\s+bound",
                "QUANTIFIED_TEMPORAL_FORMULA_WITH_ACTION_LEVEL_BOUND",
                "Quantified temporal formula with action-level bound"
            ),
            4315: (
                r"Quantification\s+with\s+temporal\s+level\s+bound",
                "QUANTIFICATION_WITH_TEMPORAL_LEVEL_BOUND",
                "Quantification with temporal-level bound"
            ),
            
            # SANY Label Errors (4330-4337)
            4330: (
                r"Label\s+parameter\s+[`']?(\w+)[`']?\s+(?:is\s+)?(?:repeated|used\s+multiple\s+times)",
                "LABEL_PARAMETER_REPETITION",
                "Label parameter repeated"
            ),
            4331: (
                r"Label\s+parameter\s+[`']?(\w+)[`']?\s+(?:is\s+)?missing",
                "LABEL_PARAMETER_MISSING",
                "Required label parameter missing"
            ),
            4332: (
                r"Label\s+parameter\s+[`']?(\w+)[`']?\s+(?:is\s+)?unnecessary",
                "LABEL_PARAMETER_UNNECESSARY",
                "Unnecessary label parameter provided"
            ),
            4333: (
                r"Label\s+[`']?(\w+)[`']?\s+(?:is\s+)?not\s+(?:in\s+)?(?:(?:a\s+)?definition\s+(?:or\s+)?(?:proof\s+)?step)",
                "LABEL_NOT_IN_DEFINITION_OR_PROOF_STEP",
                "Label not in definition or proof step"
            ),
            4334: (
                r"Label\s+(?:is\s+)?not\s+allowed\s+in\s+nested\s+ASSUME\s+PROVE\s+with\s+NEW",
                "LABEL_NOT_ALLOWED_IN_NESTED_ASSUME_PROVE_WITH_NEW",
                "Label not allowed in nested ASSUME PROVE with NEW"
            ),
            4335: (
                r"Label\s+(?:is\s+)?not\s+allowed\s+in\s+function\s+EXCEPT",
                "LABEL_NOT_ALLOWED_IN_FUNCTION_EXCEPT",
                "Label not allowed in function EXCEPT"
            ),
            4336: (
                r"Label\s+[`']?(\w+)[`']?\s+(?:is\s+)?(?:already\s+)?(?:re)?defined",
                "LABEL_REDEFINITION",
                "Label redefined"
            ),
            4337: (
                r"Label\s+[`']?(\w+)[`']?\s+(?:is\s+)?given\s+(?:the\s+)?wrong\s+number\s+of\s+arguments",
                "LABEL_GIVEN_INCORRECT_NUMBER_OF_ARGUMENTS",
                "Label given wrong number of arguments"
            ),
            
            # SANY Proof-related Errors (4350-4357)
            4350: (
                r"Proof\s+step\s+with\s+implicit\s+level\s+cannot\s+have\s+(?:a\s+)?name",
                "PROOF_STEP_WITH_IMPLICIT_LEVEL_CANNOT_HAVE_NAME",
                "Proof step with implicit level cannot have name"
            ),
            4351: (
                r"(?:Proof\s+step\s+)?(?:non-)?expression\s+used\s+(?:as\s+)?(?:an\s+)?expression|Non-expression\s+used\s+as\s+expression",
                "PROOF_STEP_NON_EXPRESSION_USED_AS_EXPRESSION",
                "Non-expression used as expression in proof step"
            ),
            4352: (
                r"Temporal\s+proof\s+goal\s+with\s+non-constant\s+(?:TAKE|WITNESS|HAVE)",
                "TEMPORAL_PROOF_GOAL_WITH_NON_CONSTANT_TAKE_WITNESS_HAVE",
                "Temporal proof goal with non-constant TAKE/WITNESS/HAVE"
            ),
            4353: (
                r"Temporal\s+proof\s+goal\s+with\s+non-constant\s+CASE",
                "TEMPORAL_PROOF_GOAL_WITH_NON_CONSTANT_CASE",
                "Temporal proof goal with non-constant CASE"
            ),
            4354: (
                r"Quantified\s+temporal\s+(?:PICK\s+)?formula\s+with\s+non-constant\s+bound",
                "QUANTIFIED_TEMPORAL_PICK_FORMULA_WITH_NON_CONSTANT_BOUND",
                "Quantified temporal PICK formula with non-constant bound"
            ),
            4355: (
                r"ASSUME\s*\/\s*PROVE\s+(?:is\s+)?used\s+where\s+(?:an\s+)?expression\s+(?:is\s+)?required",
                "ASSUME_PROVE_USED_WHERE_EXPRESSION_REQUIRED",
                "ASSUME/PROVE used where expression required"
            ),
            4356: (
                r"ASSUME\s*\/\s*PROVE\s+NEW\s+constant\s+has\s+temporal\s+level\s+bound",
                "ASSUME_PROVE_NEW_CONSTANT_HAS_TEMPORAL_LEVEL_BOUND",
                "ASSUME/PROVE NEW constant has temporal level bound"
            ),
            4357: (
                r"USE\s+or\s+HIDE\s+fact\s+(?:is\s+)?not\s+valid",
                "USE_OR_HIDE_FACT_NOT_VALID",
                "USE or HIDE fact not valid"
            ),
            
            # SANY Internal Errors (4003-4005)
            4003: (
                r"Internal\s+error",
                "INTERNAL_ERROR",
                "Internal SANY error"
            ),
            4004: (
                r"Suspected\s+unreachable\s+(?:code\s+)?check",
                "SUSPECTED_UNREACHABLE_CHECK",
                "Suspected unreachable code check failed"
            ),
            4005: (
                r"Unsupported\s+language\s+feature",
                "UNSUPPORTED_LANGUAGE_FEATURE",
                "Unsupported TLA+ language feature"
            ),
            
            # SANY Syntax/Parse Errors (Common but not in official ErrorCode.java)
            # These are parser-level errors that SANY reports but don't have official error codes
            9001: (
                r"(?:Item\s+at\s+line\s+\d+.*)?(?:is\s+)?not\s+properly\s+indented\s+inside\s+(?:conjunction|disjunction)",
                "INDENTATION_ERROR",
                "Improper indentation in conjunction/disjunction"
            ),
            9002: (
                r"Precedence\s+conflict\s+between\s+ops",
                "OPERATOR_PRECEDENCE_CONFLICT",
                "Operator precedence conflict"
            ),
            9003: (
                r"Encountered\s+[\"']([^\"']*)[\"']\s+at\s+line\s+\d+.*(?:and\s+token|column)",
                "UNEXPECTED_TOKEN",
                "Unexpected token encountered"
            ),
            9004: (
                r"Was\s+expecting\s+[\"']([^\"']*)[\"']",
                "EXPECTED_TOKEN_MISSING",
                "Expected token missing"
            ),
            9005: (
                r"(?:expressions?\s+)?(?:at\s+location\s+)?.*follow\s+each\s+other\s+without\s+any\s+intervening\s+operator",
                "MISSING_OPERATOR",
                "Missing operator between expressions"
            ),
            9006: (
                r"Fatal\s+errors?\s+while\s+parsing\s+TLA\+\s+spec",
                "GENERAL_PARSE_ERROR",
                "General TLA+ parsing error"
            ),
            
            # Special unknown error code for fallback classification
            9999: (
                r".*",  # Matches any string as last resort
                "UNKNOWN_SANY_ERROR",
                "Unmatched SANY error requiring pattern analysis"
            ),
            
            # SANY Warnings (4800+)
            4800: (
                r"Extended\s+modules?\s+symbol\s+unification\s+ambiguity",
                "EXTENDED_MODULES_SYMBOL_UNIFICATION_AMBIGUITY",
                "Symbol unification ambiguity in extended modules"
            ),
            4801: (
                r"(?:Instanced\s+)?modules?\s+symbol\s+unification\s+ambiguity",
                "INSTANCED_MODULES_SYMBOL_UNIFICATION_AMBIGUITY", 
                "Symbol unification ambiguity in instanced modules"
            ),
            4802: (
                r"Record\s+constructor\s+field\s+name\s+clash",
                "RECORD_CONSTRUCTOR_FIELD_NAME_CLASH",
                "Record constructor field name clash"
            ),
        }
        
        return patterns
    
    def classify_sany_error(self, error_message: str) -> Optional[SANYErrorMatch]:
        """
        Classify SANY error message and return matching error code.
        
        Args:
            error_message: Error message text from SANY
            
        Returns:
            SANYErrorMatch if pattern matches, None otherwise
        """
        if not error_message or not error_message.strip():
            return None
        
        # Try to match each pattern
        best_match = None
        best_confidence = 0.0
        
        for error_code, (pattern, error_name, description) in self._error_patterns.items():
            match = re.search(pattern, error_message, re.IGNORECASE | re.MULTILINE)
            if match:
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_confidence(pattern, match, error_message)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = SANYErrorMatch(
                        error_code=error_code,
                        error_name=error_name,
                        confidence=confidence,
                        matched_pattern=pattern,
                        description=description
                    )
        
        # If we found a high-confidence match, return it
        if best_match and best_confidence > 0.6:
            return best_match
        
        # FALLBACK: If no patterns matched with sufficient confidence, 
        # create an UNKNOWN error classification for debugging
        import logging
        logger = logging.getLogger(__name__)
        
        logger.warning(f"SANY Error Classification Failed - Unknown Error Pattern")
        logger.warning(f"Error message content (first 500 chars): {repr(error_message[:500])}")
        
        # Return unknown error classification to ensure we don't lose the error
        return SANYErrorMatch(
            error_code=9999,  # Special code for unknown SANY errors
            error_name="UNKNOWN_SANY_ERROR",
            confidence=0.5,  # Medium confidence since we know it's an error, just don't know the type
            matched_pattern="<unmatched>",
            description="Unmatched SANY error - needs pattern analysis"
        )
    
    def _calculate_confidence(self, pattern: str, match: re.Match, full_message: str) -> float:
        """
        Calculate confidence score for a pattern match.
        
        Args:
            pattern: Regex pattern that matched
            match: Match object
            full_message: Full error message
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        base_confidence = 0.7  # Base confidence for any match
        
        # Increase confidence for longer matches
        match_length = len(match.group(0))
        message_length = len(full_message)
        length_bonus = min(0.2, match_length / message_length)
        
        # Increase confidence for patterns with specific keywords
        specific_keywords = [
            'operator', 'function', 'module', 'symbol', 'defined', 
            'redefined', 'missing', 'incorrect', 'level', 'constraint'
        ]
        keyword_bonus = 0.0
        for keyword in specific_keywords:
            if keyword.lower() in pattern.lower():
                keyword_bonus += 0.02
        
        # Decrease confidence for very generic patterns
        generic_penalty = 0.0
        if len(pattern) < 20:  # Very short patterns are less reliable
            generic_penalty = 0.1
        
        final_confidence = min(1.0, base_confidence + length_bonus + keyword_bonus - generic_penalty)
        return final_confidence
    
    def get_error_category(self, error_code: int) -> SANYErrorCategory:
        """Get the category for a SANY error code."""
        if 4000 <= error_code <= 4002:
            return SANYErrorCategory.PARSER_INTERNAL
        elif 4003 <= error_code <= 4005:
            return SANYErrorCategory.INTERNAL_ERROR
        elif 4200 <= error_code <= 4206:
            return SANYErrorCategory.BASIC_SEMANTIC
        elif 4220 <= error_code <= 4224:
            return SANYErrorCategory.MODULE_IMPORT
        elif 4240 <= error_code <= 4247:
            return SANYErrorCategory.INSTANCE_SUBSTITUTION
        elif 4260 <= error_code <= 4262:
            return SANYErrorCategory.FUNCTION_RECORD
        elif 4270 <= error_code <= 4275:
            return SANYErrorCategory.HIGHER_ORDER
        elif 4290 <= error_code <= 4294:
            return SANYErrorCategory.RECURSIVE
        elif 4310 <= error_code <= 4315:
            return SANYErrorCategory.TEMPORAL
        elif 4330 <= error_code <= 4337:
            return SANYErrorCategory.LABEL
        elif 4350 <= error_code <= 4357:  # Updated range to match actual codes
            return SANYErrorCategory.PROOF
        elif error_code >= 9000 and error_code < 9999:  # Custom syntax/parse errors
            return SANYErrorCategory.SYNTAX_PARSE
        elif error_code == 9999:  # Special unknown error code
            return SANYErrorCategory.INTERNAL_ERROR  # Classify unknowns as internal for now
        elif error_code >= 4800:
            return SANYErrorCategory.WARNING
        else:
            return SANYErrorCategory.INTERNAL_ERROR
    
    def get_all_supported_error_codes(self) -> List[int]:
        """Get list of all SANY error codes that can be reverse-engineered."""
        return list(self._error_patterns.keys())
    
    def get_pattern_for_code(self, error_code: int) -> Optional[Tuple[str, str, str]]:
        """Get pattern, name and description for a given error code."""
        return self._error_patterns.get(error_code)


# Convenience functions
def classify_sany_error_message(error_message: str) -> Optional[SANYErrorMatch]:
    """Classify SANY error message using default classifier."""
    classifier = SANYErrorCodeReverse()
    return classifier.classify_sany_error(error_message)


def extract_sany_error_code(error_message: str) -> Optional[int]:
    """Extract SANY error code from error message."""
    match = classify_sany_error_message(error_message)
    return match.error_code if match else None