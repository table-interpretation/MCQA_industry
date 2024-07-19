import argparse
import json
from neo4j import GraphDatabase
import pandas as pd


class KGNeo4j:
    """The KG in Neo4j. Prior to running this code, make sure the following plugins
    are installed: Neosemantics(n10s) and APOC."""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_new_graph(self, turtle_file):
        with self.driver.session() as session:
            session.execute_write(self._delete_all)
            session.execute_write(self._apoc_assert)
            session.execute_write(self._create_uri_constraint)
            session.execute_write(self._configure_graph)
            session.execute_write(self._import_rdf, turtle_file)

    def close(self):
        self.driver.close()

    def run_query(self, query, params={}):
        with self.driver.session() as session:
            result = session.run(query, params)
            response = [r.values() if len(r) > 1 else r.values()[0] for r in result]
            # print(json.dumps(response, indent=4))
            # print("total results:", len(response))
        return response

    @staticmethod
    def _delete_all(tx):
        """note: make sure to install APOC plugin for Neo4j to use this procedure"""
        result = tx.run(
            'CALL apoc.periodic.commit("MATCH (a) with a limit $limit detach delete a return count(*)", {limit:10000})'
        )
        return result

    @staticmethod
    def _apoc_assert(tx):
        """drops all other existing indexes and constraints"""
        result = tx.run('CALL apoc.schema.assert({},{},true)')
        return result

    @staticmethod
    def _create_uri_constraint(tx):
        """creates constraint for uri property of Resource nodes ensuring uniqueness
        and optimal performance of queries"""
        result = tx.run('CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE')
        return result

    @staticmethod
    def _create_constraints(tx, ns, prop):
        """creates constraint for uri property of Resource nodes ensuring uniqueness
        @param ns: namespace or label of node
        @param prop: property to be unique
        """
        result = tx.run('CREATE CONSTRAINT n10s_unique_{} FOR (r:{}) REQUIRE r.{} IS UNIQUE'.format(ns, ns, prop))
        return result

    @staticmethod
    def _configure_graph(tx):
        result = tx.run("CALL n10s.graphconfig.init()")
        return result

    @staticmethod
    def _import_rdf(tx, turtle_file):
        result = tx.run("CALL n10s.rdf.import.fetch('file:///{}', 'Turtle')".format(turtle_file))
        print(json.dumps(result.values(), indent=4))
        return result

    def get_graph(self):
        # all connected nodes for the prompt
        query_prompt_graph_quant = "MATCH(tag)<-[:ns0__hasQuality]-(eq)-[:ns1__hasDatum]->(datum)-[:ns1__datumUOM]->(uom) " \
                                   "WHERE size(datum.rdfs__label) >= 1 " \
                                   "WITH * " \
                                   "MATCH(quant)-[:ns0__qualityMeasuredAs]->(datum) " \
                                   "WITH * " \
                                   "OPTIONAL MATCH(eq)-[:ns0__hasDefinition]->(def_eq) " \
                                   "WITH * " \
                                   "OPTIONAL MATCH(quant)-[:ns0__hasDefinition]->(def_quant) " \
                                   "RETURN tag.rdfs__label as Tag, eq.uri AS Equipment_URI, " \
                                   "eq.rdfs__label AS Equipment_Label, " \
                                   "def_eq.posc, " \
                                   "quant.rdfs__label AS Quant, " \
                                   "def_quant.posc, " \
                                   "datum.rdfs__label AS Datum, " \
                                   "uom.rdfs__label AS UOM " \
                                   "ORDER BY Tag;"
        return self.run_query(query_prompt_graph_quant)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        default=",",
        help="path to turtle file as input"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        default=",",
        help="path to csv file as output"
    )

    args = parser.parse_args()

    user = "neo4j"
    pw_tableml = str(input("Enter password: "))

    neo4j_db = KGNeo4j("bolt://localhost:7687", user, pw_tableml)
    neo4j_db.create_new_graph(args.input)

    # convert graph to pandas dataframe and save as csv
    df = pd.DataFrame(neo4j_db.get_graph(),
                      columns=['Tag', 'Eq_URI', 'Eq_Label', 'Eq_Def', 'Quant', 'Quant_Def', 'Datum', 'UOM'])
    df.to_csv(args.output, index=False)

    neo4j_db.close()
