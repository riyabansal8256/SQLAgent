from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import sqlite3
import pandas as pd
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime, timedelta
import hashlib
import json
from collections import deque, Counter
import time
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from dataclasses import dataclass
import pickle

# Load spaCy model for NLP
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Enhanced Data Structures
@dataclass
class QueryPattern:
    pattern: str
    sql_template: str
    confidence: float
    parameters: List[str]

@dataclass
class QueryIntent:
    intent_type: str
    entities: Dict[str, List[str]]
    modifiers: Dict[str, any]
    confidence: float

# Enhanced SQL Query Parser with Advanced NLP
class AdvancedSQLQueryParser:
    """Advanced NLP model for parsing natural language to SQL with learning capabilities"""
    
    def __init__(self):
        self.nlp = nlp
        self.query_patterns = self._initialize_patterns()
        self.learned_patterns = []
        self.entity_recognizer = EntityRecognizer()
        self.intent_classifier = IntentClassifier()
        self.sql_builder = SQLBuilder()
        
        # Initialize vectorizers for similarity matching
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
        self.pattern_vectors = None
        self.pattern_cache = {}
        
    def _initialize_patterns(self) -> List[QueryPattern]:
        """Initialize comprehensive query patterns"""
        patterns = [
            # Basic SELECT patterns
            QueryPattern(
                pattern=r"show (?:me )?all (?:the )?(.+?)(?:\s+from)?",
                sql_template="SELECT * FROM {table}",
                confidence=0.9,
                parameters=[]
            ),
            QueryPattern(
                pattern=r"(?:get|fetch|display|list) (?:all )?(.+?)(?:\s+where)?",
                sql_template="SELECT {columns} FROM {table}",
                confidence=0.85,
                parameters=['columns']
            ),
            
            # Aggregation patterns
            QueryPattern(
                pattern=r"(?:what is |find )?(?:the )?average (?:of )?(.+?)(?:\s+by)?",
                sql_template="SELECT AVG({column}) FROM {table}",
                confidence=0.9,
                parameters=['column']
            ),
            QueryPattern(
                pattern=r"count (?:the )?(?:number of )?(.+?)(?:\s+where)?",
                sql_template="SELECT COUNT({column}) FROM {table}",
                confidence=0.9,
                parameters=['column']
            ),
            QueryPattern(
                pattern=r"sum (?:of |the )?(.+?)(?:\s+for)?",
                sql_template="SELECT SUM({column}) FROM {table}",
                confidence=0.9,
                parameters=['column']
            ),
            
            # Complex patterns with conditions
            QueryPattern(
                pattern=r"(.+?) where (.+?) (?:is|equals|=) (.+)",
                sql_template="SELECT {columns} FROM {table} WHERE {condition_column} = '{condition_value}'",
                confidence=0.85,
                parameters=['columns', 'condition_column', 'condition_value']
            ),
            QueryPattern(
                pattern=r"(.+?) (?:greater|more) than (.+)",
                sql_template="SELECT * FROM {table} WHERE {column} > {value}",
                confidence=0.85,
                parameters=['column', 'value']
            ),
            
            # GROUP BY patterns
            QueryPattern(
                pattern=r"(.+?) (?:grouped |group )by (.+)",
                sql_template="SELECT {columns}, COUNT(*) FROM {table} GROUP BY {group_column}",
                confidence=0.85,
                parameters=['columns', 'group_column']
            ),
            
            # JOIN patterns
            QueryPattern(
                pattern=r"(.+?) (?:joined with|join) (.+?) on (.+)",
                sql_template="SELECT * FROM {table1} JOIN {table2} ON {join_condition}",
                confidence=0.8,
                parameters=['table1', 'table2', 'join_condition']
            ),
            
            # Top N patterns
            QueryPattern(
                pattern=r"top (\d+) (.+?)(?:\s+by)?",
                sql_template="SELECT * FROM {table} ORDER BY {order_column} DESC LIMIT {limit}",
                confidence=0.9,
                parameters=['limit', 'order_column']
            )
        ]
        return patterns
    
    def parse_query(self, text: str, table_name: str, columns: List[str]) -> Dict:
        """Enhanced query parsing with NLP and pattern matching"""
        # Preprocess text
        text_lower = text.lower().strip()
        doc = self.nlp(text_lower)
        
        # Extract intent and entities
        intent = self.intent_classifier.classify(doc, text_lower)
        entities = self.entity_recognizer.extract(doc, columns, table_name)
        
        # Build SQL query
        sql_components = self.sql_builder.build(intent, entities, table_name, columns)
        
        return sql_components
    
    def learn_pattern(self, natural_query: str, sql_query: str, success: bool):
        """Learn from successful query translations"""
        if success:
            pattern = QueryPattern(
                pattern=natural_query.lower(),
                sql_template=sql_query,
                confidence=0.7,
                parameters=self._extract_parameters(sql_query)
            )
            self.learned_patterns.append(pattern)
            
            # Update pattern cache
            self.pattern_cache[natural_query.lower()] = sql_query
    
    def _extract_parameters(self, sql_query: str) -> List[str]:
        """Extract parameter placeholders from SQL query"""
        return re.findall(r'{(\w+)}', sql_query)

# Entity Recognition Component
class EntityRecognizer:
    """Extract entities from natural language queries"""
    
    def __init__(self):
        self.column_synonyms = {
            'name': ['names', 'person', 'people', 'student', 'students'],
            'marks': ['score', 'scores', 'grade', 'grades', 'points'],
            'class': ['classes', 'course', 'courses', 'subject'],
            'section': ['sections', 'group', 'groups', 'division']
        }
        
    def extract(self, doc, columns: List[str], table_name: str) -> Dict[str, List[str]]:
        """Extract entities from parsed document"""
        entities = {
            'columns': [],
            'values': [],
            'numbers': [],
            'operators': [],
            'aggregations': []
        }
        
        # Extract columns with fuzzy matching
        for token in doc:
            token_text = token.text.lower()
            
            # Direct column match
            for col in columns:
                if self._fuzzy_match(token_text, col.lower()):
                    entities['columns'].append(col)
                    
                # Check synonyms
                col_lower = col.lower()
                if col_lower in self.column_synonyms:
                    for synonym in self.column_synonyms[col_lower]:
                        if synonym in token_text:
                            entities['columns'].append(col)
                            break
            
            # Extract numbers
            if token.like_num or token.pos_ == "NUM":
                entities['numbers'].append(token.text)
            
            # Extract quoted values
            if token.is_quote:
                entities['values'].append(token.text)
        
        # Extract values from quotes
        quoted_values = re.findall(r"['\"]([^'\"]+)['\"]", doc.text)
        entities['values'].extend(quoted_values)
        
        # Extract operators
        operator_patterns = {
            'equals': ['=', 'is', 'equals', 'equal to'],
            'greater': ['>', 'greater than', 'more than', 'above'],
            'less': ['<', 'less than', 'below', 'under'],
            'not_equal': ['!=', '<>', 'not equal', 'not'],
            'like': ['like', 'contains', 'includes'],
            'between': ['between', 'range']
        }
        
        text_lower = doc.text.lower()
        for op_type, patterns in operator_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    entities['operators'].append(op_type)
                    break
        
        # Extract aggregation functions
        agg_functions = ['count', 'sum', 'average', 'avg', 'max', 'maximum', 'min', 'minimum']
        for func in agg_functions:
            if func in text_lower:
                entities['aggregations'].append(func.upper() if func != 'average' else 'AVG')
        
        return entities
    
    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Fuzzy string matching"""
        if text1 == text2:
            return True
        
        # Calculate similarity ratio
        longer = max(len(text1), len(text2))
        shorter = min(len(text1), len(text2))
        
        if longer == 0:
            return True
            
        # Simple character-based similarity
        matches = sum(c1 == c2 for c1, c2 in zip(text1, text2))
        similarity = (matches + shorter) / (2 * longer)
        
        return similarity >= threshold

# Intent Classification Component
class IntentClassifier:
    """Classify the intent of the query"""
    
    def __init__(self):
        self.intent_keywords = {
            'select': ['show', 'display', 'get', 'fetch', 'list', 'find', 'retrieve'],
            'aggregate': ['count', 'sum', 'average', 'avg', 'max', 'min', 'total'],
            'filter': ['where', 'filter', 'with', 'having', 'only'],
            'sort': ['sort', 'order', 'arrange', 'rank'],
            'group': ['group', 'grouped', 'by', 'category', 'per'],
            'join': ['join', 'combine', 'merge', 'relate'],
            'distinct': ['unique', 'distinct', 'different'],
            'limit': ['top', 'first', 'limit', 'last']
        }
        
    def classify(self, doc, text: str) -> QueryIntent:
        """Classify query intent"""
        intents = []
        entities = {}
        modifiers = {}
        
        text_lower = text.lower()
        
        # Check for each intent type
        for intent_type, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    intents.append(intent_type)
                    break
        
        # Default to select if no intent found
        if not intents:
            intents = ['select']
        
        # Extract modifiers
        limit_match = re.search(r'(?:top|first|limit)\s+(\d+)', text_lower)
        if limit_match:
            modifiers['limit'] = int(limit_match.group(1))
        
        # Check for order direction
        if 'desc' in text_lower or 'descending' in text_lower:
            modifiers['order_direction'] = 'DESC'
        elif 'asc' in text_lower or 'ascending' in text_lower:
            modifiers['order_direction'] = 'ASC'
        
        return QueryIntent(
            intent_type=intents[0] if intents else 'select',
            entities=entities,
            modifiers=modifiers,
            confidence=0.8
        )

# SQL Query Builder Component
class SQLBuilder:
    """Build SQL queries from intent and entities"""
    
    def build(self, intent: QueryIntent, entities: Dict, table_name: str, columns: List[str]) -> Dict:
        """Build SQL components from intent and entities"""
        sql_components = {
            'action': intent.intent_type,
            'columns': entities.get('columns', []),
            'table': table_name,
            'where_clause': None,
            'group_by': None,
            'order_by': None,
            'limit': intent.modifiers.get('limit'),
            'aggregation': None,
            'distinct': False
        }
        
        # Build based on intent type
        if intent.intent_type == 'aggregate':
            if entities.get('aggregations'):
                sql_components['aggregation'] = entities['aggregations'][0]
        
        # Build WHERE clause
        if entities.get('operators') and (entities.get('columns') or entities.get('values')):
            sql_components['where_clause'] = self._build_where_clause(entities)
        
        # Build ORDER BY
        if intent.intent_type == 'sort' and entities.get('columns'):
            sql_components['order_by'] = entities['columns'][0]
            sql_components['order_direction'] = intent.modifiers.get('order_direction', 'ASC')
        
        return sql_components
    
    def _build_where_clause(self, entities: Dict) -> str:
        """Build WHERE clause from entities"""
        conditions = []
        
        columns = entities.get('columns', [])
        values = entities.get('values', [])
        numbers = entities.get('numbers', [])
        operators = entities.get('operators', ['equals'])
        
        # Combine values and numbers
        all_values = values + numbers
        
        # Build conditions
        for i, col in enumerate(columns):
            if i < len(all_values):
                op = operators[0] if operators else 'equals'
                value = all_values[i]
                
                # Format based on operator
                if op == 'equals':
                    condition = f"{col} = '{value}'"
                elif op == 'greater':
                    condition = f"{col} > {value}"
                elif op == 'less':
                    condition = f"{col} < {value}"
                elif op == 'like':
                    condition = f"{col} LIKE '%{value}%'"
                else:
                    condition = f"{col} = '{value}'"
                
                conditions.append(condition)
        
        return ' AND '.join(conditions) if conditions else None

# Enhanced Query Optimization Engine
class AdvancedQueryOptimizer:
    """Advanced query optimization with cost-based analysis"""
    
    def __init__(self):
        self.optimization_rules = {
            'index_suggestion': self.suggest_indexes,
            'join_optimization': self.optimize_joins,
            'subquery_optimization': self.optimize_subqueries,
            'predicate_pushdown': self.pushdown_predicates,
            'query_rewrite': self.rewrite_query,
            'statistics_based': self.statistics_optimization
        }
        self.query_stats = {}
        self.index_stats = {}
        
    def analyze_query(self, sql: str, table_stats: Optional[Dict] = None) -> Dict:
        """Comprehensive query analysis with cost estimation"""
        analysis = {
            'original_query': sql,
            'optimizations': [],
            'suggestions': [],
            'estimated_improvement': 0,
            'execution_plan': self._estimate_execution_plan(sql),
            'cost_estimate': 0
        }
        
        # Parse query structure
        query_structure = self._parse_query_structure(sql)
        
        # Apply optimization rules
        for rule_name, rule_func in self.optimization_rules.items():
            suggestions = rule_func(sql, query_structure, table_stats)
            if suggestions:
                analysis['optimizations'].extend(suggestions)
        
        # Calculate cost and improvement estimates
        analysis['cost_estimate'] = self._estimate_query_cost(query_structure, table_stats)
        analysis['estimated_improvement'] = self._calculate_improvement(analysis['optimizations'])
        
        return analysis
    
    def _parse_query_structure(self, sql: str) -> Dict:
        """Parse SQL query structure"""
        structure = {
            'type': 'SELECT',  # Default
            'tables': [],
            'columns': [],
            'conditions': [],
            'joins': [],
            'group_by': [],
            'order_by': [],
            'aggregations': [],
            'subqueries': []
        }
        
        sql_upper = sql.upper()
        
        # Extract query type
        if sql_upper.startswith('SELECT'):
            structure['type'] = 'SELECT'
        elif sql_upper.startswith('INSERT'):
            structure['type'] = 'INSERT'
        elif sql_upper.startswith('UPDATE'):
            structure['type'] = 'UPDATE'
        elif sql_upper.startswith('DELETE'):
            structure['type'] = 'DELETE'
        
        # Extract tables
        table_pattern = r'FROM\s+(\w+)'
        tables = re.findall(table_pattern, sql, re.IGNORECASE)
        structure['tables'] = tables
        
        # Extract WHERE conditions
        where_pattern = r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|$)'
        where_match = re.search(where_pattern, sql, re.IGNORECASE)
        if where_match:
            structure['conditions'] = self._parse_conditions(where_match.group(1))
        
        # Extract JOINs
        join_pattern = r'(INNER|LEFT|RIGHT|FULL)?\s*JOIN\s+(\w+)\s+ON\s+(.+?)(?:\s+WHERE|\s+GROUP|\s+ORDER|$)'
        joins = re.findall(join_pattern, sql, re.IGNORECASE)
        structure['joins'] = joins
        
        # Extract GROUP BY
        group_pattern = r'GROUP\s+BY\s+(.+?)(?:\s+HAVING|\s+ORDER\s+BY|$)'
        group_match = re.search(group_pattern, sql, re.IGNORECASE)
        if group_match:
            structure['group_by'] = [col.strip() for col in group_match.group(1).split(',')]
        
        # Extract ORDER BY
        order_pattern = r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|$)'
        order_match = re.search(order_pattern, sql, re.IGNORECASE)
        if order_match:
            structure['order_by'] = [col.strip() for col in order_match.group(1).split(',')]
        
        # Detect aggregations
        agg_functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
        for func in agg_functions:
            if func in sql_upper:
                structure['aggregations'].append(func)
        
        # Detect subqueries
        subquery_count = sql.count('(SELECT')
        structure['subqueries'] = ['subquery'] * subquery_count
        
        return structure
    
    def _parse_conditions(self, where_clause: str) -> List[Dict]:
        """Parse WHERE clause conditions"""
        conditions = []
        
        # Split by AND/OR
        condition_parts = re.split(r'\s+(AND|OR)\s+', where_clause, flags=re.IGNORECASE)
        
        for part in condition_parts:
            if part.upper() in ['AND', 'OR']:
                continue
                
            # Parse individual condition
            condition_match = re.match(r'(\w+)\s*([=<>!]+|LIKE|IN|BETWEEN)\s*(.+)', part, re.IGNORECASE)
            if condition_match:
                conditions.append({
                    'column': condition_match.group(1),
                    'operator': condition_match.group(2),
                    'value': condition_match.group(3)
                })
        
        return conditions
    
    def suggest_indexes(self, sql: str, structure: Dict, table_stats: Optional[Dict]) -> List[Dict]:
        """Advanced index suggestions based on query patterns"""
        suggestions = []
        
        # Analyze WHERE clause columns
        for condition in structure['conditions']:
            column = condition['column']
            operator = condition['operator']
            
            # Suggest index for equality and range queries
            if operator in ['=', '<', '>', '<=', '>=']:
                suggestions.append({
                    'type': 'index',
                    'message': f'Create index on column "{column}" for {operator} operations',
                    'impact': 'high',
                    'sql': f'CREATE INDEX idx_{column} ON {structure["tables"][0]}({column});'
                })
            
            # Special handling for LIKE queries
            elif operator.upper() == 'LIKE':
                value = condition['value'].strip("'\"")
                if not value.startswith('%'):
                    suggestions.append({
                        'type': 'index',
                        'message': f'Create index on column "{column}" for prefix LIKE queries',
                        'impact': 'medium',
                        'sql': f'CREATE INDEX idx_{column} ON {structure["tables"][0]}({column});'
                    })
        
        # Suggest composite indexes for multiple conditions
        if len(structure['conditions']) > 1:
            columns = [c['column'] for c in structure['conditions']]
            suggestions.append({
                'type': 'composite_index',
                'message': f'Consider composite index on columns: {", ".join(columns)}',
                'impact': 'high',
                'sql': f'CREATE INDEX idx_composite ON {structure["tables"][0]}({", ".join(columns)});'
            })
        
        # Join column indexes
        for join in structure['joins']:
            join_condition = join[2]
            # Extract columns from join condition
            columns = re.findall(r'(\w+\.\w+|\w+)', join_condition)
            for col in columns:
                if '.' not in col:
                    suggestions.append({
                        'type': 'join_index',
                        'message': f'Create index on join column "{col}"',
                        'impact': 'high',
                        'sql': f'CREATE INDEX idx_join_{col} ON table({col});'
                    })
        
        return suggestions
    
    def optimize_joins(self, sql: str, structure: Dict, table_stats: Optional[Dict]) -> List[Dict]:
        """Optimize JOIN operations"""
        suggestions = []
        
        if len(structure['joins']) > 1:
            suggestions.append({
                'type': 'join_order',
                'message': 'Consider reordering joins - start with smallest table',
                'impact': 'medium'
            })
        
        # Check for missing join conditions
        for join in structure['joins']:
            if not join[2].strip():
                suggestions.append({
                    'type': 'missing_join_condition',
                    'message': f'Missing JOIN condition for table {join[1]}',
                    'impact': 'critical'
                })
        
        return suggestions
    
    def optimize_subqueries(self, sql: str, structure: Dict, table_stats: Optional[Dict]) -> List[Dict]:
        """Optimize subqueries"""
        suggestions = []
        
        if structure['subqueries']:
            # Check for IN subqueries
            if ' IN (SELECT' in sql.upper():
                suggestions.append({
                    'type': 'subquery_to_join',
                    'message': 'Consider converting IN subquery to JOIN for better performance',
                    'impact': 'high',
                    'example': 'SELECT * FROM t1 JOIN t2 ON t1.id = t2.id'
                })
            
            # Check for correlated subqueries
            if len(structure['subqueries']) > 0:
                suggestions.append({
                    'type': 'correlated_subquery',
                    'message': 'Review correlated subqueries - consider using JOINs or CTEs',
                    'impact': 'high'
                })
        
        return suggestions
    
    def pushdown_predicates(self, sql: str, structure: Dict, table_stats: Optional[Dict]) -> List[Dict]:
        """Suggest predicate pushdown optimizations"""
        suggestions = []
        
        # Check if WHERE clause can be pushed down to subqueries
        if structure['subqueries'] and structure['conditions']:
            suggestions.append({
                'type': 'predicate_pushdown',
                'message': 'Consider pushing WHERE conditions into subqueries',
                'impact': 'medium'
            })
        
        return suggestions
    
    def rewrite_query(self, sql: str, structure: Dict, table_stats: Optional[Dict]) -> List[Dict]:
        """Suggest query rewrites"""
        suggestions = []
        
        # Check for SELECT *
        if 'SELECT *' in sql.upper():
            suggestions.append({
                'type': 'column_selection',
                'message': 'Specify exact columns instead of SELECT * to reduce data transfer',
                'impact': 'medium'
            })
        
        # Check for DISTINCT unnecessary usage
        if 'DISTINCT' in sql.upper() and structure['group_by']:
            suggestions.append({
                'type': 'redundant_distinct',
                'message': 'DISTINCT may be redundant with GROUP BY',
                'impact': 'low'
            })
        
        # Check for unnecessary sorting
        if structure['order_by'] and not any(word in sql.upper() for word in ['LIMIT', 'TOP']):
            suggestions.append({
                'type': 'unnecessary_sort',
                'message': 'Consider if ORDER BY is necessary without LIMIT',
                'impact': 'medium'
            })
        
        return suggestions
    
    def statistics_optimization(self, sql: str, structure: Dict, table_stats: Optional[Dict]) -> List[Dict]:
        """Optimization based on table statistics"""
        suggestions = []
        
        if table_stats:
            # Check for large table scans
            for table in structure['tables']:
                if table in table_stats and table_stats[table].get('row_count', 0) > 10000:
                    if not structure['conditions']:
                        suggestions.append({
                            'type': 'full_table_scan',
                            'message': f'Full table scan on large table "{table}" ({table_stats[table]["row_count"]} rows)',
                            'impact': 'critical'
                        })
        
        return suggestions
    
    def _estimate_query_cost(self, structure: Dict, table_stats: Optional[Dict]) -> float:
        """Estimate query execution cost"""
        cost = 0.0
        
        # Base cost for query type
        cost += {'SELECT': 1, 'INSERT': 2, 'UPDATE': 3, 'DELETE': 3}.get(structure['type'], 1)
        
        # Cost for table scans
        if table_stats:
            for table in structure['tables']:
                if table in table_stats:
                    rows = table_stats[table].get('row_count', 1000)
                    cost += np.log10(rows + 1)
        
        # Cost for joins
        cost += len(structure['joins']) * 2
        
        # Cost for subqueries
        cost += len(structure['subqueries']) * 3
        
        # Cost for sorting
        if structure['order_by']:
            cost += 1.5
        
        # Cost for grouping
        if structure['group_by']:
            cost += 1.5
        
        # Reduce cost if conditions limit rows
        if structure['conditions']:
            cost *= 0.7
        
        return cost
    
    def _calculate_improvement(self, optimizations: List[Dict]) -> float:
        """Calculate estimated improvement percentage"""
        improvement = 0.0
        
        impact_scores = {
            'critical': 30,
            'high': 20,
            'medium': 10,
            'low': 5
        }
        
        for opt in optimizations:
            improvement += impact_scores.get(opt.get('impact', 'low'), 5)
        
        return min(improvement, 90)  # Cap at 90%
    
    def _estimate_execution_plan(self, sql: str) -> List[str]:
        """Estimate query execution plan"""
        plan = []
        
        sql_upper = sql.upper()
        
        # Basic execution plan steps
        if 'FROM' in sql_upper:
            plan.append("1. Table Access")
        
        if 'WHERE' in sql_upper:
            plan.append("2. Filter Rows")
        
        if 'JOIN' in sql_upper:
            plan.append("3. Join Tables")
        
        if 'GROUP BY' in sql_upper:
            plan.append("4. Group Results")
        
        if 'HAVING' in sql_upper:
            plan.append("5. Filter Groups")
        
        if 'ORDER BY' in sql_upper:
            plan.append("6. Sort Results")
        
        if 'LIMIT' in sql_upper:
            plan.append("7. Limit Results")
        
        return plan

# Enhanced Cache Manager with Adaptive TTL
class AdaptiveCacheManager:
    """Intelligent caching with adaptive TTL and pattern learning"""
    
    def __init__(self, max_size: int = 100, default_ttl: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.query_times = {}
        self.access_patterns = {}
        self.ttl_adjustments = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0
        
    def _generate_key(self, sql: str, db_path: str) -> str:
        """Generate cache key from SQL query and database"""
        combined = f"{db_path}:{sql}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, sql: str, db_path: str) -> Optional[Tuple[List, float]]:
        """Get cached result with adaptive TTL adjustment"""
        key = self._generate_key(sql, db_path)
        
        if key in self.cache:
            timestamp, data, ttl = self.cache[key]
            
            # Check if cache entry has expired
            if datetime.now() - timestamp < timedelta(seconds=ttl):
                self.hit_count += 1
                self.access_times[key] = datetime.now()
                
                # Update access pattern
                self._update_access_pattern(key)
                
                # Adjust TTL based on access frequency
                self._adjust_ttl(key)
                
                return data, self.query_times.get(key, 0)
            else:
                # Remove expired entry
                self._evict_entry(key)
        
        self.miss_count += 1
        return None
    
    def set(self, sql: str, db_path: str, data: List, query_time: float, ttl: Optional[int] = None):
        """Cache query result with adaptive TTL"""
        key = self._generate_key(sql, db_path)
        
        # Evict least valuable entry if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_least_valuable()
        
        # Calculate adaptive TTL
        ttl = ttl or self._calculate_adaptive_ttl(key, query_time)
        
        # Store in cache
        self.cache[key] = (datetime.now(), data, ttl)
        self.query_times[key] = query_time
        self.access_times[key] = datetime.now()
        
        # Initialize access pattern
        if key not in self.access_patterns:
            self.access_patterns[key] = deque(maxlen=10)
        self.access_patterns[key].append(datetime.now())
    
    def _calculate_adaptive_ttl(self, key: str, query_time: float) -> int:
        """Calculate TTL based on query complexity and access patterns"""
        base_ttl = self.default_ttl
        
        # Adjust based on query execution time
        if query_time > 1.0:  # Expensive queries
            base_ttl *= 2
        elif query_time < 0.1:  # Fast queries
            base_ttl *= 0.5
        
        # Adjust based on access frequency
        if key in self.access_patterns and len(self.access_patterns[key]) > 3:
            # Frequently accessed - increase TTL
            frequency_factor = min(len(self.access_patterns[key]) / 5, 2.0)
            base_ttl *= frequency_factor
        
        # Apply any manual TTL adjustments
        if key in self.ttl_adjustments:
            base_ttl *= self.ttl_adjustments[key]
        
        return int(base_ttl)
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for adaptive behavior"""
        if key not in self.access_patterns:
            self.access_patterns[key] = deque(maxlen=10)
        self.access_patterns[key].append(datetime.now())
    
    def _adjust_ttl(self, key: str):
        """Dynamically adjust TTL based on access patterns"""
        if key in self.access_patterns and len(self.access_patterns[key]) >= 2:
            # Calculate average time between accesses
            times = list(self.access_patterns[key])
            intervals = [(times[i+1] - times[i]).total_seconds() 
                        for i in range(len(times)-1)]
            
            if intervals:
                avg_interval = np.mean(intervals)
                
                # If accessed frequently, increase TTL
                if avg_interval < 60:  # Less than 1 minute
                    self.ttl_adjustments[key] = 1.5
                elif avg_interval < 300:  # Less than 5 minutes
                    self.ttl_adjustments[key] = 1.2
                else:
                    self.ttl_adjustments[key] = 1.0
    
    def _evict_least_valuable(self):
        """Evict least valuable cache entry using scoring algorithm"""
        if not self.cache:
            return
        
        scores = {}
        current_time = datetime.now()
        
        for key, (timestamp, _, ttl) in self.cache.items():
            # Calculate value score
            age = (current_time - timestamp).total_seconds()
            last_access = (current_time - self.access_times.get(key, timestamp)).total_seconds()
            query_time = self.query_times.get(key, 0.1)
            access_count = len(self.access_patterns.get(key, []))
            
            # Score formula: higher is more valuable
            score = (query_time * 100) + (access_count * 10) - (age / 60) - (last_access / 30)
            scores[key] = score
        
        # Evict entry with lowest score
        evict_key = min(scores, key=scores.get)
        self._evict_entry(evict_key)
    
    def _evict_entry(self, key: str):
        """Remove entry from all cache structures"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.query_times.pop(key, None)
        self.access_patterns.pop(key, None)
        self.ttl_adjustments.pop(key, None)
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()
        self.query_times.clear()
        self.access_patterns.clear()
        self.ttl_adjustments.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'avg_query_time': np.mean(list(self.query_times.values())) if self.query_times else 0
        }

# Enhanced Performance Monitor
class PerformanceMonitor:
    """Advanced performance monitoring with analytics"""
    
    def __init__(self):
        self.query_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_queries': 0,
            'total_time': 0,
            'slow_queries': [],
            'query_patterns': Counter(),
            'error_queries': [],
            'optimization_suggestions': []
        }
        self.thresholds = {
            'slow_query_ms': 100,
            'very_slow_query_ms': 500
        }
    
    def record_query(self, natural_query: str, sql_query: str, execution_time: float, 
                    result_count: int, success: bool = True, error: str = None):
        """Record query execution details"""
        query_record = {
            'timestamp': datetime.now(),
            'natural_query': natural_query,
            'sql_query': sql_query,
            'execution_time': execution_time,
            'result_count': result_count,
            'success': success,
            'error': error
        }
        
        self.query_history.append(query_record)
        self._update_metrics(query_record)
        self._analyze_patterns(query_record)
    
    def _update_metrics(self, record: Dict):
        """Update performance metrics"""
        self.performance_metrics['total_queries'] += 1
        self.performance_metrics['total_time'] += record['execution_time']
        
        # Track slow queries
        if record['execution_time'] > self.thresholds['slow_query_ms'] / 1000:
            self.performance_metrics['slow_queries'].append({
                'query': record['sql_query'],
                'time': record['execution_time'],
                'timestamp': record['timestamp']
            })
        
        # Track errors
        if not record['success']:
            self.performance_metrics['error_queries'].append({
                'query': record['natural_query'],
                'error': record['error'],
                'timestamp': record['timestamp']
            })
        
        # Update query patterns
        query_type = self._extract_query_type(record['sql_query'])
        self.performance_metrics['query_patterns'][query_type] += 1
    
    def _analyze_patterns(self, record: Dict):
        """Analyze query patterns for optimization suggestions"""
        # Check for repeated slow queries
        sql_query = record['sql_query']
        slow_count = sum(1 for q in self.performance_metrics['slow_queries'] 
                        if q['query'] == sql_query)
        
        if slow_count >= 3:
            suggestion = {
                'type': 'repeated_slow_query',
                'query': sql_query,
                'message': f'This query has been slow {slow_count} times. Consider optimization.',
                'timestamp': datetime.now()
            }
            if suggestion not in self.performance_metrics['optimization_suggestions']:
                self.performance_metrics['optimization_suggestions'].append(suggestion)
    
    def _extract_query_type(self, sql_query: str) -> str:
        """Extract query type for pattern analysis"""
        sql_upper = sql_query.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            if 'JOIN' in sql_upper:
                return 'SELECT_JOIN'
            elif 'GROUP BY' in sql_upper:
                return 'SELECT_AGGREGATE'
            elif 'WHERE' in sql_upper:
                return 'SELECT_FILTER'
            else:
                return 'SELECT_SIMPLE'
        elif sql_upper.startswith('INSERT'):
            return 'INSERT'
        elif sql_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif sql_upper.startswith('DELETE'):
            return 'DELETE'
        else:
            return 'OTHER'
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if self.performance_metrics['total_queries'] == 0:
            return {'message': 'No queries recorded yet'}
        
        avg_time = self.performance_metrics['total_time'] / self.performance_metrics['total_queries']
        
        # Calculate percentiles
        execution_times = [q['execution_time'] for q in self.query_history if q['success']]
        percentiles = {}
        if execution_times:
            percentiles = {
                'p50': np.percentile(execution_times, 50),
                'p90': np.percentile(execution_times, 90),
                'p95': np.percentile(execution_times, 95),
                'p99': np.percentile(execution_times, 99)
            }
        
        return {
            'total_queries': self.performance_metrics['total_queries'],
            'avg_execution_time': avg_time,
            'percentiles': percentiles,
            'slow_queries_count': len(self.performance_metrics['slow_queries']),
            'error_count': len(self.performance_metrics['error_queries']),
            'query_patterns': dict(self.performance_metrics['query_patterns']),
            'recent_slow_queries': self.performance_metrics['slow_queries'][-5:],
            'optimization_suggestions': self.performance_metrics['optimization_suggestions'][-5:]
        }
    
    def get_query_history(self, limit: int = 50) -> List[Dict]:
        """Get recent query history"""
        return list(self.query_history)[-limit:]
    
    def clear_history(self):
        """Clear all history and metrics"""
        self.query_history.clear()
        self.performance_metrics = {
            'total_queries': 0,
            'total_time': 0,
            'slow_queries': [],
            'query_patterns': Counter(),
            'error_queries': [],
            'optimization_suggestions': []
        }

# Database Helper Functions
def create_database(file_path: str, db_path: str) -> Dict:
    """Create SQLite database from Excel file with metadata"""
    df = pd.read_excel(file_path)
    conn = sqlite3.connect(db_path)
    
    # Create main table
    table_name = "data_table"
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    
    # Create metadata table
    metadata = {
        'table_name': table_name,
        'columns': df.columns.tolist(),
        'row_count': len(df),
        'column_types': df.dtypes.to_dict(),
        'creation_time': datetime.now().isoformat()
    }
    
    # Store metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS table_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    for key, value in metadata.items():
        conn.execute("INSERT OR REPLACE INTO table_metadata (key, value) VALUES (?, ?)",
                    (key, json.dumps(value) if isinstance(value, (list, dict)) else str(value)))
    
    conn.commit()
    conn.close()
    
    return metadata

def get_table_info(db_path: str) -> Dict:
    """Get comprehensive table information"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]
    
    table_info = {}
    for table in tables:
        if table == 'table_metadata':
            continue
            
        # Get column info
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
        sample_data = cursor.fetchall()
        
        table_info[table] = {
            'columns': [col[1] for col in columns],
            'column_types': {col[1]: col[2] for col in columns},
            'row_count': row_count,
            'sample_data': sample_data
        }
    
    conn.close()
    return table_info

def execute_query(sql: str, db_path: str) -> Tuple[List, List, float]:
    """Execute SQL query with timing"""
    start_time = time.time()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql)
        columns = [description[0] for description in cursor.description] if cursor.description else []
        results = cursor.fetchall()
        execution_time = time.time() - start_time
        
        return results, columns, execution_time
    finally:
        conn.close()

# Initialize Components
@st.cache_resource
def init_components():
    """Initialize all components"""
    return {
        'parser': AdvancedSQLQueryParser(),
        'optimizer': AdvancedQueryOptimizer(),
        'cache': AdaptiveCacheManager(),
        'monitor': PerformanceMonitor()
    }

# Main Streamlit Application
def main():
    st.set_page_config(page_title="Advanced SQL Query Agent", layout="wide")
    
    st.title("🚀 Advanced SQL Query Agent")
    st.markdown("Natural Language to SQL with AI-Powered Optimization")
    
    # Initialize components
    components = init_components()
    parser = components['parser']
    optimizer = components['optimizer']
    cache = components['cache']
    monitor = components['monitor']
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Feature toggles
        use_cache = st.checkbox("Enable Query Caching", value=True)
        use_optimization = st.checkbox("Enable Query Optimization", value=True)
        show_performance = st.checkbox("Show Performance Metrics", value=True)
        
        # Cache controls
        if use_cache:
            st.subheader("Cache Settings")
            if st.button("Clear Cache"):
                cache.clear()
                st.success("Cache cleared!")
            
            cache_stats = cache.get_stats()
            st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.2%}")
            st.metric("Cache Size", cache_stats['size'])
    
    # File upload or sample data
    st.header("📊 Data Source")
    data_source = st.radio("Choose data source:", ["Upload Excel File", "Use Sample Data"])
    
    if data_source == "Upload Excel File":
        uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_db:
                db_path = tmp_db.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                metadata = create_database(tmp_file.name, db_path)
                st.success(f"Database created with {metadata['row_count']} rows")
    else:
        # Create sample database
        sample_data = pd.DataFrame({
            'student_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'marks': [85, 92, 78, 95, 88],
            'class': ['A', 'B', 'A', 'B', 'A'],
            'section': [1, 2, 1, 2, 1]
        })
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_db:
            db_path = tmp_db.name
            
        conn = sqlite3.connect(db_path)
        sample_data.to_sql('students', conn, index=False, if_exists='replace')
        conn.close()
        
        st.success("Sample database created")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query", "📈 Performance", "🔧 Optimization", "📊 Visualization"])
    
    with tab1:
        st.header("Natural Language Query")
        
        # Get table info
        if 'db_path' in locals():
            table_info = get_table_info(db_path)
            
            if table_info:
                table_name = list(table_info.keys())[0]
                columns = table_info[table_name]['columns']
                
                # Display table info
                with st.expander("📋 Table Information"):
                    st.write(f"**Table:** {table_name}")
                    st.write(f"**Columns:** {', '.join(columns)}")
                    st.write(f"**Rows:** {table_info[table_name]['row_count']}")
                
                # Query input
                natural_query = st.text_area(
                    "Enter your question in natural language:",
                    placeholder="e.g., Show me all students with marks greater than 80"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    generate_sql = st.button("Generate SQL", type="primary")
                with col2:
                    execute_sql = st.button("Execute Query")
                
                if generate_sql and natural_query:
                    with st.spinner("Generating SQL..."):
                        # Parse natural language to SQL
                        try:
                            sql_components = parser.parse_query(natural_query, table_name, columns)
                            
                            # Build SQL query
                            sql_parts = ["SELECT"]
                            
                            # Add DISTINCT if needed
                            if sql_components.get('distinct'):
                                sql_parts.append("DISTINCT")
                            
                            # Add columns or aggregation
                            if sql_components.get('aggregation'):
                                agg_func = sql_components['aggregation']
                                col = sql_components['columns'][0] if sql_components['columns'] else '*'
                                sql_parts.append(f"{agg_func}({col})")
                            else:
                                cols = sql_components['columns'] if sql_components['columns'] else ['*']
                                sql_parts.append(', '.join(cols))
                            
                            # Add FROM clause
                            sql_parts.append(f"FROM {sql_components['table']}")
                            
                            # Add WHERE clause
                            if sql_components.get('where_clause'):
                                sql_parts.append(f"WHERE {sql_components['where_clause']}")
                            
                            # Add GROUP BY
                            if sql_components.get('group_by'):
                                sql_parts.append(f"GROUP BY {sql_components['group_by']}")
                            
                            # Add ORDER BY
                            if sql_components.get('order_by'):
                                direction = sql_components.get('order_direction', 'ASC')
                                sql_parts.append(f"ORDER BY {sql_components['order_by']} {direction}")
                            
                            # Add LIMIT
                            if sql_components.get('limit'):
                                sql_parts.append(f"LIMIT {sql_components['limit']}")
                            
                            generated_sql = ' '.join(sql_parts)
                            st.session_state['generated_sql'] = generated_sql
                            
                            st.success("SQL query generated!")
                            st.code(generated_sql, language='sql')
                            
                            # Show optimization suggestions if enabled
                            if use_optimization:
                                analysis = optimizer.analyze_query(generated_sql, table_info)
                                
                                if analysis['optimizations']:
                                    st.warning("🔧 Optimization Suggestions:")
                                    for opt in analysis['optimizations']:
                                        st.write(f"- {opt['message']}")
                                        if 'sql' in opt:
                                            st.code(opt['sql'], language='sql')
                            
                        except Exception as e:
                            st.error(f"Error generating SQL: {str(e)}")
                
                if execute_sql and 'generated_sql' in st.session_state:
                    sql = st.session_state['generated_sql']
                    
                    # Check cache first
                    cached_result = None
                    if use_cache:
                        cached_result = cache.get(sql, db_path)
                    
                    if cached_result:
                        results, query_time = cached_result
                        st.info("📦 Result from cache")
                    else:
                        # Execute query
                        try:
                            results, columns, query_time = execute_query(sql, db_path)
                            
                            # Cache the result
                            if use_cache:
                                cache.set(sql, db_path, (results, columns), query_time)
                            
                        except Exception as e:
                            st.error(f"Query execution error: {str(e)}")
                            monitor.record_query(natural_query, sql, 0, 0, False, str(e))
                            return
                    
                    # Record in performance monitor
                    monitor.record_query(natural_query, sql, query_time, len(results))
                    
                    # Display results
                    st.success(f"Query executed in {query_time*1000:.2f} ms")
                    
                    if results:
                        df_results = pd.DataFrame(results, columns=columns if 'columns' in locals() else None)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Download option
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="Download results as CSV",
                            data=csv,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No results found")
    
    with tab2:
        if show_performance:
            st.header("📈 Performance Analytics")
            
            report = monitor.get_performance_report()
            
            if report.get('message'):
                st.info(report['message'])
            else:
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Queries", report['total_queries'])
                with col2:
                    st.metric("Avg Execution Time", f"{report['avg_execution_time']*1000:.2f} ms")
                with col3:
                    st.metric("Slow Queries", report['slow_queries_count'])
                with col4:
                    st.metric("Errors", report['error_count'])
                
                # Query patterns
                if report['query_patterns']:
                    st.subheader("Query Pattern Distribution")
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=list(report['query_patterns'].keys()),
                            values=list(report['query_patterns'].values()),
                            hole=0.3
                        )
                    ])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance percentiles
                if report.get('percentiles'):
                    st.subheader("Performance Percentiles")
                    
                    percentiles_df = pd.DataFrame([
                        {"Percentile": k, "Time (ms)": v*1000} 
                        for k, v in report['percentiles'].items()
                    ])
                    
                    fig = px.bar(percentiles_df, x="Percentile", y="Time (ms)",
                                title="Query Execution Time Percentiles")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recent slow queries
                if report['recent_slow_queries']:
                    st.subheader("Recent Slow Queries")
                    for query in report['recent_slow_queries']:
                        with st.expander(f"Query ({query['time']*1000:.0f} ms)"):
                            st.code(query['query'], language='sql')
                            st.write(f"Executed at: {query['timestamp']}")
                
                # Query history
                st.subheader("Recent Query History")
                history = monitor.get_query_history(20)
                
                if history:
                    history_df = pd.DataFrame([
                        {
                            "Time": h['timestamp'].strftime("%H:%M:%S"),
                            "Natural Query": h['natural_query'][:50] + "..." if len(h['natural_query']) > 50 else h['natural_query'],
                            "Execution Time (ms)": h['execution_time']*1000,
                            "Results": h['result_count'],
                            "Success": "✅" if h['success'] else "❌"
                        }
                        for h in history
                    ])
                    st.dataframe(history_df, use_container_width=True)
    
    with tab3:
        st.header("🔧 Query Optimization Analysis")
        
        # Manual query input for optimization
        manual_sql = st.text_area(
            "Enter SQL query to analyze:",
            placeholder="SELECT * FROM table WHERE column = 'value'"
        )
        
        if st.button("Analyze Query"):
            if manual_sql and 'db_path' in locals():
                table_info = get_table_info(db_path)
                analysis = optimizer.analyze_query(manual_sql, table_info)
                
                # Display analysis results
                st.subheader("Query Analysis Results")
                
                # Execution plan
                st.write("**Estimated Execution Plan:**")
                for step in analysis['execution_plan']:
                    st.write(f"  {step}")
                
                # Cost estimate
                st.metric("Estimated Query Cost", f"{analysis['cost_estimate']:.2f}")
                st.metric("Potential Improvement", f"{analysis['estimated_improvement']:.0f}%")
                
                # Optimization suggestions
                if analysis['optimizations']:
                    st.subheader("Optimization Recommendations")
                    
                    for opt in analysis['optimizations']:
                        impact_color = {
                            'critical': '🔴',
                            'high': '🟠',
                            'medium': '🟡',
                            'low': '🟢'
                        }
                        
                        with st.expander(f"{impact_color.get(opt.get('impact', 'low'), '⚪')} {opt['type'].replace('_', ' ').title()}"):
                            st.write(opt['message'])
                            if 'sql' in opt:
                                st.code(opt['sql'], language='sql')
                            if 'example' in opt:
                                st.write("**Example:**")
                                st.code(opt['example'], language='sql')
                else:
                    st.success("No optimization issues found!")
        
        # Learning from feedback
        st.subheader("📚 Query Pattern Learning")
        
        col1, col2 = st.columns(2)
        with col1:
            feedback_natural = st.text_input("Natural language query:")
        with col2:
            feedback_sql = st.text_input("Correct SQL query:")
        
        if st.button("Teach Pattern"):
            if feedback_natural and feedback_sql:
                parser.learn_pattern(feedback_natural, feedback_sql, True)
                st.success("Pattern learned! The system will use this in future translations.")
    
    with tab4:
        st.header("📊 Data Visualization")
        
        if 'generated_sql' in st.session_state and 'db_path' in locals():
            st.info("Execute a query first to visualize the results")
            
            # Get last query results
            if st.button("Visualize Last Query Results"):
                sql = st.session_state['generated_sql']
                
                try:
                    results, columns, _ = execute_query(sql, db_path)
                    
                    if results:
                        df = pd.DataFrame(results, columns=columns)
                        
                        # Auto-detect visualization type
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
                        
                        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                            # Bar chart
                            st.subheader("Bar Chart")
                            x_col = st.selectbox("X-axis (Category):", categorical_cols)
                            y_col = st.selectbox("Y-axis (Value):", numeric_cols)
                            
                            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if len(numeric_cols) >= 2:
                            # Scatter plot
                            st.subheader("Scatter Plot")
                            x_col = st.selectbox("X-axis:", numeric_cols, key='scatter_x')
                            y_col = st.selectbox("Y-axis:", numeric_cols, key='scatter_y')
                            
                            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if len(categorical_cols) >= 1:
                            # Pie chart for distribution
                            st.subheader("Distribution")
                            cat_col = st.selectbox("Category:", categorical_cols, key='pie_cat')
                            
                            value_counts = df[cat_col].value_counts()
                            fig = go.Figure(data=[
                                go.Pie(labels=value_counts.index, values=value_counts.values)
                            ])
                            fig.update_layout(title=f"Distribution of {cat_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Time series if datetime column exists
                        datetime_cols = []
                        for col in df.columns:
                            try:
                                pd.to_datetime(df[col])
                                datetime_cols.append(col)
                            except:
                                pass
                        
                        if datetime_cols and numeric_cols:
                            st.subheader("Time Series")
                            time_col = st.selectbox("Time column:", datetime_cols)
                            value_col = st.selectbox("Value column:", numeric_cols, key='ts_value')
                            
                            df[time_col] = pd.to_datetime(df[time_col])
                            df_sorted = df.sort_values(time_col)
                            
                            fig = px.line(df_sorted, x=time_col, y=value_col, 
                                        title=f"{value_col} over {time_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Heatmap for correlation
                        if len(numeric_cols) >= 2:
                            st.subheader("Correlation Heatmap")
                            
                            corr_matrix = df[numeric_cols].corr()
                            
                            fig = go.Figure(data=go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                colorscale='RdBu',
                                zmid=0,
                                text=corr_matrix.values.round(2),
                                texttemplate='%{text}',
                                textfont={"size": 10},
                                hoverongaps=False
                            ))
                            
                            fig.update_layout(
                                title="Feature Correlation Matrix",
                                xaxis_nticks=len(numeric_cols),
                                yaxis_nticks=len(numeric_cols),
                                width=700,
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Custom visualization
                        st.subheader("Custom Visualization")
                        
                        viz_type = st.selectbox(
                            "Choose visualization type:",
                            ["Box Plot", "Histogram", "Violin Plot", "3D Scatter", "Sunburst"]
                        )
                        
                        if viz_type == "Box Plot" and numeric_cols and categorical_cols:
                            cat_col = st.selectbox("Category:", categorical_cols, key='box_cat')
                            num_col = st.selectbox("Values:", numeric_cols, key='box_num')
                            
                            fig = px.box(df, x=cat_col, y=num_col, 
                                       title=f"Distribution of {num_col} by {cat_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "Histogram" and numeric_cols:
                            hist_col = st.selectbox("Column:", numeric_cols, key='hist_col')
                            bins = st.slider("Number of bins:", 10, 50, 20)
                            
                            fig = px.histogram(df, x=hist_col, nbins=bins,
                                             title=f"Distribution of {hist_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "Violin Plot" and numeric_cols and categorical_cols:
                            cat_col = st.selectbox("Category:", categorical_cols, key='violin_cat')
                            num_col = st.selectbox("Values:", numeric_cols, key='violin_num')
                            
                            fig = px.violin(df, x=cat_col, y=num_col,
                                          title=f"{num_col} Distribution by {cat_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "3D Scatter" and len(numeric_cols) >= 3:
                            x_col = st.selectbox("X-axis:", numeric_cols, key='3d_x')
                            y_col = st.selectbox("Y-axis:", numeric_cols, key='3d_y')
                            z_col = st.selectbox("Z-axis:", numeric_cols, key='3d_z')
                            
                            color_col = None
                            if categorical_cols:
                                use_color = st.checkbox("Color by category")
                                if use_color:
                                    color_col = st.selectbox("Color:", categorical_cols)
                            
                            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                                              color=color_col,
                                              title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == "Sunburst" and len(categorical_cols) >= 2:
                            path_cols = st.multiselect(
                                "Select hierarchy (order matters):",
                                categorical_cols,
                                default=categorical_cols[:2]
                            )
                            
                            if len(path_cols) >= 2:
                                value_col = None
                                if numeric_cols:
                                    use_values = st.checkbox("Use values for sizing")
                                    if use_values:
                                        value_col = st.selectbox("Value column:", numeric_cols)
                                
                                fig = px.sunburst(df, path=path_cols, values=value_col,
                                                title="Hierarchical Data Visualization")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Export visualizations
                        st.subheader("Export Options")
                        
                        export_format = st.selectbox(
                            "Export format:",
                            ["PNG", "SVG", "HTML", "JSON"]
                        )
                        
                        if st.button("Generate Export"):
                            if 'fig' in locals():
                                if export_format == "PNG":
                                    img_bytes = fig.to_image(format="png")
                                    st.download_button(
                                        label="Download PNG",
                                        data=img_bytes,
                                        file_name=f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                        mime="image/png"
                                    )
                                elif export_format == "SVG":
                                    img_str = fig.to_image(format="svg").decode()
                                    st.download_button(
                                        label="Download SVG",
                                        data=img_str,
                                        file_name=f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                                        mime="image/svg+xml"
                                    )
                                elif export_format == "HTML":
                                    html_str = fig.to_html()
                                    st.download_button(
                                        label="Download HTML",
                                        data=html_str,
                                        file_name=f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                        mime="text/html"
                                    )
                                elif export_format == "JSON":
                                    json_str = fig.to_json()
                                    st.download_button(
                                        label="Download JSON",
                                        data=json_str,
                                        file_name=f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json"
                                    )
                
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
    
    # Footer with statistics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cache Hit Rate", f"{cache.get_stats()['hit_rate']:.1%}")
    
    with col2:
        perf_report = monitor.get_performance_report()
        avg_time = perf_report.get('avg_execution_time', 0) * 1000
        st.metric("Avg Query Time", f"{avg_time:.1f} ms")
    
    with col3:
        st.metric("Total Queries", perf_report.get('total_queries', 0))
    
    with col4:
        st.metric("Active Cache Items", cache.get_stats()['size'])
    
    # Advanced features info
    with st.expander("ℹ️ Advanced Features"):
        st.markdown("""
        ### 🧠 Custom NLP Model
        - **Entity Recognition**: Automatically identifies table names, columns, and values
        - **Intent Classification**: Understands query intent (select, aggregate, filter, etc.)
        - **Pattern Learning**: Learns from user feedback to improve translations
        
        ### ⚡ Query Optimization
        - **Cost-Based Analysis**: Estimates query execution cost
        - **Index Recommendations**: Suggests indexes for better performance
        - **Query Rewriting**: Automatically rewrites queries for optimization
        - **Join Optimization**: Recommends efficient join strategies
        
        ### 💾 Adaptive Caching
        - **Smart TTL**: Adjusts cache duration based on access patterns
        - **Value-Based Eviction**: Keeps most valuable queries in cache
        - **Performance Tracking**: Monitors cache effectiveness
        
        ### 📊 Performance Monitoring
        - **Query Analytics**: Tracks execution times and patterns
        - **Slow Query Detection**: Identifies and alerts on slow queries
        - **Pattern Analysis**: Recognizes common query patterns
        - **Optimization Suggestions**: Provides actionable improvements
        """)
    
    # Cleanup temporary files on session end
    if st.button("🗑️ Clean Up Session", help="Remove temporary files and clear cache"):
        cache.clear()
        monitor.clear_history()
        
        # Clean up any temporary database files
        if 'db_path' in locals() and os.path.exists(db_path):
            try:
                os.unlink(db_path)
                st.success("Session cleaned up successfully!")
            except Exception as e:
                st.warning(f"Could not remove temporary files: {str(e)}")

# Run the application
if __name__ == "__main__":
    main()