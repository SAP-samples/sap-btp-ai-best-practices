import { ChangeDetectionStrategy, Component, Input, OnChanges } from '@angular/core';
import { TableComponent, TableColumnComponent, TableInitialStateDirective } from '@fundamental-ngx/platform/table';
import { IconModule } from '@fundamental-ngx/core/icon';
import { WorkCenterSchedule } from '../../models/schedule.models';
import { formatDisoDate } from '../../utils/format';

interface DeferredRow {
  _work_center: string;
  ORDER_NO: string;
  ORDER_TYPE_CODE: string;
  EQUIPMENT_DESC: string;
  PRIORITY: string;
  EQUIPMENT_CRITICALITY: string;
  ACTIVITY_WORK_INVOLVE: number;
  BASIC_START_DATE: string | null;
  LATEST_EXECTN_FINISH_DATE: string | null;
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
  selector: 'app-deferred-table',
  templateUrl: './deferred-table.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [TableComponent, TableColumnComponent, TableInitialStateDirective, IconModule],
})
export class DeferredTableComponent implements OnChanges {
  @Input() schedule: WorkCenterSchedule[] = [];

  rows: DeferredRow[] = [];

  ngOnChanges(): void {
    const result: DeferredRow[] = [];
    for (const wc of this.schedule) {
      for (const op of wc.unscheduled) {
        result.push({
          _work_center: wc.work_center,
          ORDER_NO: op['ORDER_NO'] as string,
          ORDER_TYPE_CODE: op['ORDER_TYPE_CODE'] as string,
          EQUIPMENT_DESC: op['EQUIPMENT_DESC'] as string,
          PRIORITY: op['PRIORITY'] as string,
          EQUIPMENT_CRITICALITY: op['EQUIPMENT_CRITICALITY'] as string,
          ACTIVITY_WORK_INVOLVE: op['ACTIVITY_WORK_INVOLVE'] as number,
          BASIC_START_DATE: formatDisoDate(op['BASIC_START_DATE']),
          LATEST_EXECTN_FINISH_DATE: formatDisoDate(op['LATEST_EXECTN_FINISH_DATE']),
          OVERDUE: overdueLabel(op['BASIC_START_DATE']),
        });
      }
    }
    this.rows = result;
  }
}
