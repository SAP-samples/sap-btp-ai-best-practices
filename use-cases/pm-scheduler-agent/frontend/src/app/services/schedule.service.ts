import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  AiTextResponse,
  ConfirmResponse,
  DownEquipmentResult,
  MetaResponse,
  OperationRow,
  OpportunityResponse,
  ScheduleRequest,
  ScheduleResponse,
  WorkCenterSchedule,
} from '../models/schedule.models';

@Injectable({ providedIn: 'root' })
export class ScheduleService {
  private readonly http = inject(HttpClient);

  getMeta(): Observable<MetaResponse> {
    return this.http.get<MetaResponse>('/api/meta');
  }

  /** Raw order-operation detail (pre-planning), optionally scoped to a plant. */
  getOrders(plant: string, limit = 500): Observable<{ rows: OperationRow[]; total: number }> {
    let params = new HttpParams().set('limit', String(limit));
    if (plant) params = params.set('plant', plant);
    return this.http.get<{ rows: OperationRow[]; total: number }>('/api/orders', { params });
  }

  generateSchedule(req: ScheduleRequest): Observable<ScheduleResponse> {
    return this.http.post<ScheduleResponse>('/api/schedule', req);
  }

  getOpportunityOrders(
    equipmentNo: string,
    weekStart: string,
    plant: string
  ): Observable<OpportunityResponse> {
    return this.http.post<OpportunityResponse>('/api/schedule/opportunity', {
      equipment_no: equipmentNo,
      week_start: weekStart,
      plant,
    });
  }

  getOpportunityBatch(
    equipmentNos: string[],
    weekStart: string,
    plant: string
  ): Observable<{ equipment: DownEquipmentResult[] }> {
    return this.http.post<{ equipment: DownEquipmentResult[] }>(
      '/api/schedule/opportunity/batch',
      { equipment_nos: equipmentNos, week_start: weekStart, plant }
    );
  }

  aiExplain(
    schedule: WorkCenterSchedule[],
    weekStart: string,
    oppAdded: OperationRow[]
  ): Observable<AiTextResponse> {
    return this.http.post<AiTextResponse>('/api/ai/explain', {
      schedule,
      week_start: weekStart,
      opp_added: oppAdded,
    });
  }

  /** Write the scheduled dates back to S/4HANA (pin operation constraint dates). */
  confirmSchedule(schedule: WorkCenterSchedule[]): Observable<ConfirmResponse> {
    return this.http.post<ConfirmResponse>('/api/schedule/confirm', { schedule });
  }

  exportCsv(schedule: WorkCenterSchedule[], weekStart: string): Observable<Blob> {
    return this.http.post('/api/schedule/export', { schedule, week_start: weekStart }, {
      responseType: 'blob',
    });
  }

  updateBasicStart(orderNo: string, operNo: string, basicStartDate: string): Observable<{ success: boolean }> {
    return this.http.post<{ success: boolean }>('/api/orders/update-basic-start', {
      order_no: orderNo,
      oper_no: operNo,
      basic_start_date: basicStartDate,
    });
  }

  sendEmail(
    schedule: WorkCenterSchedule[],
    weekStart: string,
    plant: string,
    to: string
  ): Observable<{ success: boolean }> {
    return this.http.post<{ success: boolean }>('/api/schedule/email', {
      schedule, week_start: weekStart, plant, to,
    });
  }
}
