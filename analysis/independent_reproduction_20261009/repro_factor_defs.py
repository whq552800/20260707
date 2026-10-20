"""因素定义，供审计和确认版流程复用。2026-10-10起环境PM2.5改用GBD来源 gbd_ambient_pm25_ug_m3（覆盖203国、1990–2023年全部年份），不再使用WDI来源；repro_factor.py 为复现旧包保留WDI。"""
LOGIT=['basic_water_pct','basic_sanitation_pct','clean_cooking_pct','dtp3_pct','hib3_pct','pcv3_pct','stunting_u5_pct','low_birthweight_pct','undernourishment_pct','uhc_service_index','secondhand_smoke_sev']
LOG=['fertility_births_per_woman','health_expenditure_pc_current_usd','income_ppp_pc','gbd_ambient_pm25_ug_m3']
LOG1P=['hospital_beds_per1000','physicians_per1000','nurses_midwives_per1000','secondary_enrolment_pct']
ALL=LOGIT+LOG+LOG1P
LOWER_BETTER=['stunting_u5_pct','low_birthweight_pct','undernourishment_pct','gbd_ambient_pm25_ug_m3','secondhand_smoke_sev','fertility_births_per_woman']
CONTEXT=['income_ppp_pc','secondary_enrolment_pct','fertility_births_per_woman']
