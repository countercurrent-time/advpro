import ast
import astunparse
import random

class MutationStrategy:
    def apply(self, node):
        pass

class IdentifierRenamer(MutationStrategy):
    def apply(self, node, target_identifier):
        class Renamer(ast.NodeTransformer):
            def __init__(self, target_identifier):
                self.target_identifier = target_identifier

            def visit_Name(self, node):
                if node.id == self.target_identifier:
                    new_id = f"renamed_{self.target_identifier}"
                    return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)
                return node

        return Renamer(target_identifier).visit(node)

class ExpressionMutator(MutationStrategy):
    def apply(self, node, target_expression):
        class Mutator(ast.NodeTransformer):
            def __init__(self, target_expression):
                self.target_expression = target_expression

            def visit_Expr(self, node):
                if ast.dump(node) == self.target_expression:
                    return ast.copy_location(ast.Expr(value=ast.Num(n=0)), node)
                return node

        return Mutator(target_expression).visit(node)

class StatementInserter(MutationStrategy):
    def apply(self, node, target_statement):
        class Inserter(ast.NodeTransformer):
            def __init__(self, target_statement):
                self.target_statement = target_statement

            def visit_Module(self, node):
                body = list(node.body)
                body.append(ast.Expr(value=ast.Name(id='pass', ctx=ast.Load())))
                return ast.Module(body=body)

        return Inserter(target_statement).visit(node)

def get_mutation_strategies():
    return [
        IdentifierRenamer,
        ExpressionMutator,
        StatementInserter,
    ]

def apply_strategies(ast_root, strategies, importance_scores):
    sorted_indices = torch.argsort(importance_scores, descending=True)[:3]
    top_tokens = [ast_root.body[i] for i in sorted_indices]

    mutated_ast = ast_root
    for token in top_tokens:
        strategy_class = random.choice(strategies)
        strategy = strategy_class()
        mutated_ast = strategy.apply(mutated_ast, token)
    return mutated_ast