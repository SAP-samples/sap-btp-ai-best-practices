import {
  ChangeDetectionStrategy,
  Component,
  Input,
  OnChanges,
  inject,
  signal,
} from '@angular/core';
import { TableComponent, TableColumnComponent, TableInitialStateDirective } from '@fundamental-ngx/platform/table';
import { OperationRow } from '../../models/schedule.models';
import { ScheduleService } from '../../services/schedule.service';
import { formatDisoDate } from '../../utils/format';

/**
 * Read-only detail view of every maintenance order-operation for a plant,
 * shown *before* any scheduling is applied. Loads lazily whenever the bound
 * plant changes (the tab is created/destroyed by the parent @switch).
 */
@Component({
  selector: 'app-data-table',
  templateUrl: './data-table.component.html',
  styleUrl: './data-table.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [TableComponent, TableColumnComponent, TableInitialStateDirective],
})
export class DataTableComponent implements OnChanges {
  @Input() plant = '';

  private readonly service = inject(ScheduleService);

  readonly DISPLAY_LIMIT = 500;

  rows = signal<OperationRow[]>([]);
  total = signal(0);
  isLoading = signal(false);
  error = signal<string | null>(null);

  ngOnChanges(): void {
    this.load();
  }

  private load(): void {
    this.isLoading.set(true);
    this.error.set(null);
    this.service.getOrders(this.plant, this.DISPLAY_LIMIT).subscribe({
      next: ({ rows, total }) => {
        this.total.set(total);
        this.rows.set(
          rows.map((r) => ({
            ...r,
            OPER_NO:
              r['OPER_NO'] === null || r['OPER_NO'] === undefined || r['OPER_NO'] === ''
                ? '—'
                : String(r['OPER_NO']).replace(/\.0+$/, ''),
            SYSTEM_STATUS:
              r['SYSTEM_STATUS'] === null || r['SYSTEM_STATUS'] === undefined
                ? ''
                : String(r['SYSTEM_STATUS'])
                    .split('\n')
                    .map((s) => s.trim())
                    .filter(Boolean)
                    .join(' · '),
            BASIC_START_DATE: formatDisoDate(r['BASIC_START_DATE']),
            BASIC_FINISH_DATE: formatDisoDate(r['BASIC_FINISH_DATE']),
            LATEST_EXECTN_FINISH_DATE: formatDisoDate(r['LATEST_EXECTN_FINISH_DATE']),
            RELEASE_DATE: formatDisoDate(r['RELEASE_DATE']),
          }))
        );
        this.isLoading.set(false);
      },
      error: (err) => {
        this.error.set(err?.error?.error ?? 'Failed to load orders.');
        this.isLoading.set(false);
      },
    });
  }
}
