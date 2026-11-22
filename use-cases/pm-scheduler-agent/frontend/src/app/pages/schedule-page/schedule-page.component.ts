import {
  ChangeDetectionStrategy,
  Component,
  OnInit,
  computed,
  inject,
  signal,
} from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { DynamicPageModule } from '@fundamental-ngx/core/dynamic-page';
import { BreadcrumbModule } from '@fundamental-ngx/core/breadcrumb';
import { ButtonModule } from '@fundamental-ngx/core/button';
import { ToolbarModule } from '@fundamental-ngx/core/toolbar';
import { IconModule } from '@fundamental-ngx/core/icon';
import { BarModule } from '@fundamental-ngx/core/bar';
import { LinkModule } from '@fundamental-ngx/core/link';
import { CheckboxComponent } from '@fundamental-ngx/core/checkbox';

import { ScheduleService } from '../../services/schedule.service';
import { markdownToHtml } from '../../utils/format';
import {
  ConfirmResponse,
  DownEquipmentResult,
  MetaResponse,
  OperationRow,
  ScheduleResponse,
  WorkCenterSchedule,
} from '../../models/schedule.models';
import { ScheduleTableComponent, BasicStartChangeEvent } from '../../components/schedule-table/schedule-table.component';
import { DeferredTableComponent } from '../../components/deferred-table/deferred-table.component';
import { DataTableComponent } from '../../components/data-table/data-table.component';
import { formatDisoDate } from '../../utils/format';

type TabId = 'schedule' | 'data' | 'deferred';

@Component({
  selector: 'app-schedule-page',
  templateUrl: './schedule-page.component.html',
  styleUrl: './schedule-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    FormsModule,
    DecimalPipe,
    DynamicPageModule,
    BreadcrumbModule,
    ButtonModule,
    ToolbarModule,
    IconModule,
    BarModule,
    LinkModule,
    CheckboxComponent,
    ScheduleTableComponent,
    DeferredTableComponent,
    DataTableComponent,
  ],
})
export class SchedulePageComponent implements OnInit {
  private readonly service = inject(ScheduleService);
  private readonly sanitizer = inject(DomSanitizer);

  activeTab = signal<TabId>('schedule');

  // Meta
  meta = signal<MetaResponse | null>(null);
  metaError = signal<string | null>(null);

  // Filter form (signals so derived state stays reactive under OnPush)
  plant = signal('');
  weekStart = signal('');
  releasedOnly = signal(true);
  priorityMedium = signal(true);
  priorityLow = signal(true);

  // Work-center selection (signal of a map for checkbox binding)
  wcSelections = signal<Record<string, boolean>>({});

  // Whether saved favorites exist for the current plant
  hasFavorites = signal(false);

  // Computed from meta + plant
  availableWorkCenters = computed(() => {
    const m = this.meta();
    return m ? (m.work_centers_by_plant[this.plant()] ?? []) : [];
  });

  availableEquipment = computed(() => {
    const m = this.meta();
    return m ? (m.equipment_by_plant[this.plant()] ?? []) : [];
  });

  selectedWorkCenters = computed(() => {
    const sel = this.wcSelections();
    return Object.keys(sel).filter((k) => sel[k]);
  });

  priorityFilter = computed(() => {
    const p: string[] = [];
    if (this.priorityMedium()) p.push('Medium');
    if (this.priorityLow()) p.push('Low');
    return p;
  });

  // Schedule result
  scheduleResult = signal<ScheduleResponse | null>(null);
  isLoading = signal(false);
  error = signal<string | null>(null);

  // Opportunistic orders added to schedule
  oppAdded = signal<OperationRow[]>([]);

  // Per-order Basic Start overrides (user edits saved to HANA)
  basicStartOverrides = signal<Record<string, string>>({});
  basicStartSaving = signal<string | null>(null);   // rowKey currently saving

  onBasicStartChange(ev: BasicStartChangeEvent): void {
    // Optimistic update
    this.basicStartOverrides.update(prev => ({ ...prev, [ev.rowKey]: ev.newDate }));
    // Persist to HANA
    this.basicStartSaving.set(ev.rowKey);
    this.service.updateBasicStart(ev.orderNo, ev.operNo, ev.newDate).subscribe({
      next: () => this.basicStartSaving.set(null),
      error: (err) => {
        console.error('HANA update failed', err);
        this.basicStartSaving.set(null);
      },
    });
  }

  // Down equipment auto-detection
  downEquipment = signal<DownEquipmentResult[]>([]);
  downEquipmentLoading = signal(false);
  downSelectedOrders = signal<Set<string>>(new Set());

  toggleDownOrder(orderNo: string): void {
    this.downSelectedOrders.update((prev) => {
      const next = new Set(prev);
      next.has(orderNo) ? next.delete(orderNo) : next.add(orderNo);
      return next;
    });
  }

  addDownOrders(): void {
    const allOrders = this.downEquipment().flatMap((e) => e.orders);
    const toAdd = allOrders.filter((o) =>
      this.downSelectedOrders().has(o['ORDER_NO'] as string)
    ).map((o) => ({
      ...o,
      BASIC_START_DATE: formatDisoDate(o['BASIC_START_DATE']),
    }));
    this.oppAdded.update((prev) => [...prev, ...toAdd]);
    this.downSelectedOrders.set(new Set());
  }

  private loadDownEquipment(res: { schedule: WorkCenterSchedule[]; week_start: string; plant: string }): void {
    const equipNos = [...new Set(
      res.schedule.flatMap((wc) =>
        wc.scheduled.map((op) => String(op['EQUIPMENT_NO']).split('.')[0])
      ).filter(Boolean)
    )];
    if (!equipNos.length) return;
    this.downEquipmentLoading.set(true);
    this.service.getOpportunityBatch(equipNos, res.week_start, res.plant).subscribe({
      next: (r) => {
        this.downEquipment.set(r.equipment);
        this.downEquipmentLoading.set(false);
      },
      error: () => this.downEquipmentLoading.set(false),
    });
  }

  // AI Explanation popup
  showAiModal = signal(false);
  aiExplanationText = signal('');
  isExplainingInPage = signal(false);
  aiExplainError = signal<string | null>(null);
  aiExplanationHtml = computed((): SafeHtml =>
    this.sanitizer.bypassSecurityTrustHtml(markdownToHtml(this.aiExplanationText()))
  );

  openAiExplanation(): void {
    if (!this.effectiveSchedule().length) return;
    this.showAiModal.set(true);
    if (this.aiExplanationText()) return; // already fetched, just show it
    this.isExplainingInPage.set(true);
    this.aiExplainError.set(null);
    this.service.aiExplain(this.effectiveSchedule(), this.weekStart(), this.oppAdded()).subscribe({
      next: (res) => {
        this.aiExplanationText.set(res.text);
        this.isExplainingInPage.set(false);
      },
      error: (err) => {
        this.aiExplainError.set(err?.error?.error ?? 'AI request failed.');
        this.isExplainingInPage.set(false);
      },
    });
  }

  // Email sending
  recipientEmail = signal('');
  isSendingEmail = signal(false);
  emailFeedback = signal<{ type: 'success' | 'error'; text: string } | null>(null);

  // Whether the Generate button should be enabled
  canGenerate = computed(
    () =>
      !this.isLoading() &&
      !!this.plant() &&
      !!this.weekStart() &&
      this.selectedWorkCenters().length > 0 &&
      this.priorityFilter().length > 0
  );

  // Derived schedule with opportunistic additions + user date overrides
  effectiveSchedule = computed((): WorkCenterSchedule[] => {
    const base = this.scheduleResult()?.schedule ?? [];
    const extras = this.oppAdded();
    const overrides = this.basicStartOverrides();

    const applyOverrides = (ops: OperationRow[]): OperationRow[] =>
      ops.map(op => {
        const operNo = String(op['OPER_NO'] ?? '').replace(/\.0+$/, '');
        const key = `${op['ORDER_NO']}_${operNo}`;
        return overrides[key] ? { ...op, BASIC_START_DATE: overrides[key] } : op;
      });

    const withOverrides = base.map(wc => ({
      ...wc,
      scheduled: applyOverrides(wc.scheduled),
    }));

    if (!extras.length) return withOverrides;

    const merged = withOverrides.map((wc) => ({ ...wc, scheduled: [...wc.scheduled] }));
    for (const op of extras) {
      const wc = op['OPER_WORK_CENTER'] as string;
      let entry = merged.find((m) => m.work_center === wc);
      if (!entry) {
        entry = {
          work_center: wc,
          capacity_available: 0,
          capacity_used: 0,
          load_pct: 0,
          scheduled: [],
          unscheduled: [],
        };
        merged.push(entry);
      }
      entry.scheduled.push({ ...op, _OPPORTUNISTIC: true });
    }
    return merged;
  });

  ngOnInit(): void {
    this.service.getMeta().subscribe({
      next: (meta) => {
        const firstPlant = meta.plants[0] ?? '';
        this.plant.set(firstPlant);
        this.weekStart.set(this.defaultWeekStart(meta.date_range));

        // Set meta last so the computeds see the correct plant on first run.
        this.meta.set(meta);
        this.resetWcSelections();
      },
      error: (err) => {
        this.metaError.set(err?.error?.error ?? 'Failed to load metadata.');
      },
    });
  }

  /**
   * Default to the Monday of the current week, clamped to the capacity
   * calendar's [min, max] range so the first "Generate" lands on data.
   */
  private defaultWeekStart(range: { min: string; max: string } | undefined): string {
    const fmt = (d: Date) =>
      `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(
        d.getDate()
      ).padStart(2, '0')}`;

    const today = new Date();
    const monday = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    monday.setDate(monday.getDate() - ((monday.getDay() + 6) % 7)); // back to Monday

    const min = range?.min ? new Date(range.min + 'T00:00:00') : null;
    const max = range?.max ? new Date(range.max + 'T00:00:00') : null;
    let ws = monday;
    if (min && ws < min) ws = min;
    if (max && ws > max) ws = max;
    return fmt(ws);
  }

  onPlantChange(plant: string): void {
    this.plant.set(plant);
    this.resetWcSelections();
  }

  setWorkCenter(wc: string, checked: boolean): void {
    this.wcSelections.update((prev) => ({ ...prev, [wc]: checked }));
  }

  saveWcFavorites(): void {
    const plant = this.plant();
    const selected = this.selectedWorkCenters();
    localStorage.setItem(`fmi_wc_fav_${plant}`, JSON.stringify(selected));
    this.hasFavorites.set(true);
  }

  clearWcFavorites(): void {
    const plant = this.plant();
    localStorage.removeItem(`fmi_wc_fav_${plant}`);
    this.hasFavorites.set(false);
    // Re-enable all work centers for this plant
    const next: Record<string, boolean> = {};
    for (const wc of this.availableWorkCenters()) next[wc] = true;
    this.wcSelections.set(next);
  }

  private resetWcSelections(): void {
    const plant = this.plant();
    const saved = localStorage.getItem(`fmi_wc_fav_${plant}`);
    const favorites: string[] = saved ? (JSON.parse(saved) as string[]) : [];
    const hasFav = favorites.length > 0;
    this.hasFavorites.set(hasFav);

    const next: Record<string, boolean> = {};
    for (const wc of this.availableWorkCenters()) {
      next[wc] = hasFav ? favorites.includes(wc) : true;
    }
    this.wcSelections.set(next);
  }

  generateSchedule(): void {
    if (!this.canGenerate()) return;
    this.isLoading.set(true);
    this.error.set(null);
    this.scheduleResult.set(null);
    this.oppAdded.set([]);
    this.aiExplanationText.set('');
    this.downEquipment.set([]);
    this.downSelectedOrders.set(new Set());
    this.basicStartOverrides.set({});
    this.service
      .generateSchedule({
        plant: this.plant(),
        work_centers: this.selectedWorkCenters(),
        week_start: this.weekStart(),
        priority_filter: this.priorityFilter(),
        released_only: this.releasedOnly(),
      })
      .subscribe({
        next: (res) => {
          this.scheduleResult.set(res);
          this.isLoading.set(false);
          this.loadDownEquipment(res);
        },
        error: (err) => {
          this.error.set(err?.error?.error ?? 'Failed to generate schedule.');
          this.isLoading.set(false);
        },
      });
  }

  // Confirm / write-back to S/4HANA
  isConfirming = signal(false);
  confirmFeedback = signal<{ type: 'success' | 'error'; text: string } | null>(null);

  confirmSchedule(): void {
    const schedule = this.effectiveSchedule();
    if (!schedule.length || this.isConfirming()) return;
    this.isConfirming.set(true);
    this.confirmFeedback.set(null);
    this.service.confirmSchedule(schedule).subscribe({
      next: (res: ConfirmResponse) => {
        this.isConfirming.set(false);
        if (res.failed.length) {
          this.confirmFeedback.set({
            type: 'error',
            text: `Updated ${res.updated}/${res.total} operations in S/4 — ${res.failed.length} failed.`,
          });
        } else {
          this.confirmFeedback.set({
            type: 'success',
            text: `Updated all ${res.updated} operations in S/4HANA.`,
          });
        }
      },
      error: (err) => {
        this.isConfirming.set(false);
        this.confirmFeedback.set({
          type: 'error',
          text: err?.error?.error ?? 'Failed to update S/4HANA.',
        });
      },
    });
  }

  exportCsv(): void {
    const schedule = this.effectiveSchedule();
    if (!schedule.length) return;
    const week = this.scheduleResult()?.week_start ?? this.weekStart();
    this.service.exportCsv(schedule, week).subscribe((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `schedule_${week}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  onOppOrdersAdded(orders: OperationRow[]): void {
    this.oppAdded.update((prev) => [...prev, ...orders]);
  }

  sendEmail(): void {
    const schedule = this.effectiveSchedule();
    const to = this.recipientEmail().trim();
    if (!schedule.length || !to) return;
    this.isSendingEmail.set(true);
    this.emailFeedback.set(null);
    const week = this.scheduleResult()?.week_start ?? this.weekStart();
    const plant = this.scheduleResult()?.plant ?? this.plant();
    this.service.sendEmail(schedule, week, plant, to).subscribe({
      next: () => {
        this.isSendingEmail.set(false);
        this.emailFeedback.set({ type: 'success', text: `Email sent to ${to}` });
      },
      error: (err) => {
        this.isSendingEmail.set(false);
        this.emailFeedback.set({
          type: 'error',
          text: err?.error?.error ?? 'Error al enviar el email.',
        });
      },
    });
  }
}
