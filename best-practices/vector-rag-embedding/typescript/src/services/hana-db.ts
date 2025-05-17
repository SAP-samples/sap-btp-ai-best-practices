import * as hana from "@sap/hana-client";
import { logger } from "../utils/logger";
const PromiseModule = require("@sap/hana-client/extension/Promise.js");

export default class HANADb {
  private readonly connectionOptions: hana.ConnectionOptions;

  constructor() {
    // Get Connection Parameters from VCAP_SERVICES
    if (process.env.VCAP_SERVICES) {
      const hanaBinding = JSON.parse(process.env.VCAP_SERVICES).hana[0];
      this.connectionOptions = {
        host: hanaBinding.credentials.host,
        port: hanaBinding.credentials.port,
        user: hanaBinding.credentials.user,
        password: hanaBinding.credentials.password,
        currentSchema: hanaBinding.credentials.schema,
        encrypt: true,
        sslTrustStore: hanaBinding.credentials.certificate,
      };
      logger.info("SAP HANA binding found.");
    } else {
      throw new Error("SAP HANA binding not found");
    }
  }

  /**
   * Executes a query on the SAP HANA database.
   * @param query The SQL query to execute.
   * @param data Optional data to bind to the query.
   * @returns The results of the query.
   */
  public async executeQuery(query: string, data?: any[]): Promise<any> {
    const hanaConnection: hana.Connection = hana.createConnection();

    try {
      // Connect to the database
      await PromiseModule.connect(hanaConnection, this.connectionOptions);

      // Execute the query
      const stmt: hana.Statement = await PromiseModule.prepare(hanaConnection, query);
      const results: hana.ResultSet = await PromiseModule.execute(stmt, data);

      // Close connection
      PromiseModule.disconnect(hanaConnection);

      return results;
    } catch (err: any) {
      logger.error(err);
    }
  }

  /**
   * Executes a query on the SAP HANA database.
   * @param query The SQL query to execute.
   * @param data Optional data to bind to the query.
   * @returns The results of the query.
   */
  public async executeBatchQuery(query: string, data?: any[]): Promise<any> {
    let result: any;
    const hanaConnection = hana.createConnection();

    try {
      // Connect to the database
      await PromiseModule.connect(hanaConnection, this.connectionOptions);

      // Execute the query
      const stmt = await PromiseModule.prepare(hanaConnection, query);
      const results = await PromiseModule.executeBatch(stmt, data);

      PromiseModule.disconnect(hanaConnection);

      return results;
    } catch (err: any) {
      logger.error(err);
    }
  }
}
