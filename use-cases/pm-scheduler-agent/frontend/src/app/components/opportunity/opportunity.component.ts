import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { TableComponent, TableColumnComponent, TableInitialStateDirective, FilterableColumnDataType } from '@fundamental-ngx/platform/table';
import { ButtonModule } from '@fundamental-ngx/core/button';
import { CheckboxComponent } from '@fundamental-ngx/core/checkbox';
import { EquipmentItem, OperationRow } from '../../models/schedule.models';
import { ScheduleService } from '../../services/schedule.service';
import { inject } from '@angular/core';
import { formatDisoDate } from '../../utils/format';

@Component({
  selector: 'app-opportunity',
  templateUrl: './opportunity.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, ButtonModule, CheckboxComponent, TableComponent, TableColumnComponent, TableInitialStateDirective],
})
export class OpportunityComponent implements OnChanges {
  @Input() equipment: EquipmentItem[] = [];
  @Input() plant = '';
  @Input() weekStart = '';
  @Output() ordersAdded = new EventEmitter<OperationRow[]>();

  private readonly service = inject(ScheduleService);

  selectedEquipmentId = '';
  // Raw rows drive selection + emit-to-schedule (dates preserved as-is).
  oppRows: OperationRow[] = [];
  // Display copy used by the table with the date column formatted for reading.
  oppDisplayRows: OperationRow[] = [];
  selectedOrderNos = new Set<string>();
  isLoading = signal(false);
  error = signal<string | null>(null);

  ngOnChanges(): void {
    this.selectedEquipmentId = '';
    this.oppRows = [];
    this.oppDisplayRows = [];
    this.selectedOrderNos.clear();
  }

  onEquipmentChange(): void {
    if (!this.selectedEquipmentId) {
      this.oppRows = [];
      this.oppDisplayRows = [];
      return;
    }
    this.isLoading.set(true);
    this.error.set(null);
    this.service
      .getOpportunityOrders(this.selectedEquipmentId, this.weekStart, this.plant)
      .subscribe({
        next: (res) => {
          this.oppRows = res.orders;
          this.oppDisplayRows = res.orders.map((r) => ({
            ...r,
            BASIC_START_DATE: formatDisoDate(r['BASIC_START_DATE']),
          }));
          this.selectedOrderNos.clear();
          this.isLoading.set(false);
        },
        error: (err) => {
          this.error.set(err?.error?.error ?? 'Failed to load opportunity orders.');
          this.isLoading.set(false);
        },
      });
  }

  toggleOrder(orderNo: string): void {
    if (this.selectedOrderNos.has(orderNo)) {
      this.selectedOrderNos.delete(orderNo);
    } else {
      this.selectedOrderNos.add(orderNo);
    }
  }

  addToSchedule(): void {
    const toAdd = this.oppRows.filter((o) => this.selectedOrderNos.has(o['ORDER_NO'] as string));
    this.ordersAdded.emit(toAdd);
    this.selectedOrderNos.clear();
  }
}
