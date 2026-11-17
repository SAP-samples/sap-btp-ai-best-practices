export interface MetaResponse {
  plants: string[];
  work_centers_by_plant: Record<string, string[]>;
  equipment_by_plant: Record<string, EquipmentItem[]>;
  date_range: { min: string; max: string };
}

export interface EquipmentItem {
  id: string;
  label: string;
}

export interface ScheduleRequest {
  plant: string;
  work_centers: string[];
  week_start: string;
  priority_filter: string[];
  released_only: boolean;
}

export interface OperationRow {
  ORDER_NO: string;
  ORDER_TYPE_CODE: string;
  OPER_SHORT_TEXT: string;
  EQUIPMENT_DESC: string;
  PRIORITY: string;
  EQUIPMENT_CRITICALITY: string;
  ACTIVITY_WORK_INVOLVE: number;
  BASIC_START_DATE: string | null;
  BASIC_FINISH_DATE: string | null;
  LATEST_EXECTN_FINISH_DATE: string | null;
  OPER_WORK_CENTER: string;
  MAINT_ACTIVITY_TYPE: string;
  WEEK_BUCKET?: string;
  _OPPORTUNISTIC?: boolean;
  [key: string]: unknown;
}

export interface WorkCenterSchedule {
  work_center: string;
  capacity_available: number;
  capacity_used: number;
  load_pct: number;
  scheduled: OperationRow[];
  unscheduled: OperationRow[];
}

export interface ScheduleResponse {
  week_start: string;
  plant: string;
  schedule: WorkCenterSchedule[];
}

export interface OpportunityResponse {
  orders: OperationRow[];
  week_buckets: string[];
}

export interface AiTextResponse {
  text: string;
}

export interface DownEquipmentResult {
  equipment_no: string;
  equipment_desc: string;
  orders: OperationRow[];
}

export interface ConfirmResponse {
  total: number;
  updated: number;
  failed: { order_no: string; oper_no: string; error?: string }[];
}
