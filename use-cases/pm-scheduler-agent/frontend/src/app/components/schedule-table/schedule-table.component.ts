import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { TableComponent, TableColumnComponent, TableInitialStateDirective, FdpCellDef } from '@fundamental-ngx/platform/table';
import { ButtonModule } from '@fundamental-ngx/core/button';
import { IconModule } from '@fundamental-ngx/core/icon';
import { WorkCenterSchedule } from '../../models/schedule.models';
import { formatDisoDate } from '../../utils/format';

export interface BasicStartChangeEvent {
  rowKey: string;
  orderNo: string;
  operNo: string;
  newDate: string;
}

interface ScheduleRow {
  _rowKey: string;
  _work_center: string;
  _load_pct: number;
  _isLinked: boolean;
  ORDER_NO: string;
  OPER_NO: string;
  OPER_SHORT_TEXT: string;
  ORDER_TYPE_CODE: string;
  EQUIPMENT_DESC: string;
  PRIORITY: string;
  EQUIPMENT_CRITICALITY: string;
  ACTIVITY_WORK_INVOLVE: number;
  SCHED_WEEK: string;
  RELEASE_DATE: string;
  BASIC_START_DATE: string | null;
  BASIC_FINISH_DATE: string | null;
  OVERDUE: string;
  [key: string]: unknown;
}

const TODAY = new Date().toISOString().slice(0, 10);

function overdueLabel(isoDate: unknown): string {
  if (!isoDate) return '';
  const d = String(isoDate).slice(0, 10);
  return d < TODAY ? '⚠ Overdue' : '';
}

@Component({
  selector: 'app-schedule-table',
  templateUrl: './schedule-table.component.html',
  styleUrl: './schedule-table.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, TableComponent, TableColumnComponent, TableInitialStateDirective, FdpCellDef, ButtonModule, IconModule],
})
export class ScheduleTableComponent implements OnChanges {
  @Input() schedule: WorkCenterSchedule[] = [];
  @Input() weekStart = '';
  @Output() basicStartChange = new EventEmitter<BasicStartChangeEvent>();

  rows: ScheduleRow[] = [];

  /** Dynamic height keeps the header sticky while the page is scrolled. */
  readonly tableBodyHeight = 'calc(100vh - 460px)';

  // Inline date-edit state
  editingKey = signal<string | null>(null);
  editingDate = signal('');

  openEdit(row: ScheduleRow): void {
    this.editingKey.set(row._rowKey);
    this.editingDate.set(row.BASIC_START_DATE ?? '');
  }

  saveEdit(row: ScheduleRow): void {
    const newDate = this.editingDate();
    if (!newDate) return;
    this.basicStartChange.emit({ rowKey: row._rowKey, orderNo: row.ORDER_NO, operNo: row.OPER_NO, newDate });
    this.editingKey.set(null);
  }

  cancelEdit(): void {
    this.editingKey.set(null);
  }

  ngOnChanges(_changes: SimpleChanges): void {
    // Identify orders that appear in more than one work center (multi-crew coordinated jobs)
    const wcByOrder = new Map<string, Set<string>>();
    for (const wc of this.schedule) {
      for (const op of wc.scheduled) {
        const oNo = String(op['ORDER_NO']);
        if (!wcByOrder.has(oNo)) wcByOrder.set(oNo, new Set());
        wcByOrder.get(oNo)!.add(wc.work_center);
      }
    }
    const linkedOrders = new Set<string>(
      [...wcByOrder.entries()].filter(([, wcs]) => wcs.size > 1).map(([oNo]) => oNo)
    );

    const result: ScheduleRow[] = [];
    for (const wc of this.schedule) {
      for (const op of wc.scheduled) {
        const operNo = op['OPER_NO'] !== null && op['OPER_NO'] !== undefined
          ? String(op['OPER_NO']).replace(/\.0+$/, '') : '—';
        const rowKey = `${op['ORDER_NO']}_${operNo}`;
        result.push({
          _rowKey: rowKey,
          _work_center: wc.work_center,
          _load_pct: wc.load_pct,
          _isLinked: linkedOrders.has(String(op['ORDER_NO'])),
          ORDER_NO: op['ORDER_NO'] as string,
          OPER_NO: operNo,
          OPER_SHORT_TEXT: (op['OPER_SHORT_TEXT'] as string) ?? '',
          ORDER_TYPE_CODE: op['ORDER_TYPE_CODE'] as string,
          EQUIPMENT_DESC: op['EQUIPMENT_DESC'] as string,
          PRIORITY: op['PRIORITY'] as string,
          EQUIPMENT_CRITICALITY: op['EQUIPMENT_CRITICALITY'] as string,
          ACTIVITY_WORK_INVOLVE: op['ACTIVITY_WORK_INVOLVE'] as number,
          SCHED_WEEK: this.weekStart,
          RELEASE_DATE: formatDisoDate(op['RELEASE_DATE']),
          BASIC_START_DATE: formatDisoDate(op['BASIC_START_DATE']),
          BASIC_FINISH_DATE: formatDisoDate(op['BASIC_FINISH_DATE']),
          OVERDUE: overdueLabel(op['BASIC_START_DATE']),
        });
      }
    }
    this.rows = result;
  }
}
