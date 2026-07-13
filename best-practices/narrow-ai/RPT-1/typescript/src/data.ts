export const classificationPredictionData = {
  prediction_config: {
    target_columns: [
      {
        name: "COSTCENTER" as const,
        prediction_placeholder: "[PREDICT]",
        task_type: "classification" as const,
      },
    ],
  },
  index_column: "ID",
  rows: [
    {
      PRODUCT: "Couch",
      PRICE: 999.99,
      ORDERDATE: "2025-11-28",
      ID: "35",
      COSTCENTER: "[PREDICT]",
    },
    {
      PRODUCT: "Office Chair",
      PRICE: 150.8,
      ORDERDATE: "2025-11-02",
      ID: "44",
      COSTCENTER: "Office Furniture",
    },
    {
      PRODUCT: "Server Rack",
      PRICE: 2200.0,
      ORDERDATE: "2025-11-01",
      ID: "104",
      COSTCENTER: "Data Infrastructure",
    },
  ],
};

export const discountDataSchema = [
  { name: "PRODUCT", dtype: "string" },
  { name: "PRICE", dtype: "numeric" },
  { name: "ORDERDATE", dtype: "date" },
  { name: "ID", dtype: "string" },
  { name: "DISCOUNT_RATE", dtype: "numeric" },
] as const;

export const discountPredictionData = {
  prediction_config: {
    target_columns: [
      {
        name: "DISCOUNT_RATE" as const,
        prediction_placeholder: -1,
        task_type: "regression" as const,
      },
    ],
  },
  index_column: "ID" as const,
  parse_data_types: true,
  rows: [
    {
      PRODUCT: "Standing Desk",
      PRICE: 780.0,
      ORDERDATE: "2025-11-30" as const,
      ID: "88",
      DISCOUNT_RATE: -1,
    },
    {
      PRODUCT: "Office Chair",
      PRICE: 150.8,
      ORDERDATE: "2025-11-02" as const,
      ID: "44",
      DISCOUNT_RATE: 0.05,
    },
    {
      PRODUCT: "Server Rack",
      PRICE: 2200.0,
      ORDERDATE: "2025-11-01" as const,
      ID: "104",
      DISCOUNT_RATE: 0.12,
    },
    {
      PRODUCT: "Monitor Arm",
      PRICE: 115.2,
      ORDERDATE: "2025-11-08" as const,
      ID: "52",
      DISCOUNT_RATE: 0.04,
    },
  ],
};
