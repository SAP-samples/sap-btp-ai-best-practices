@path    : 'orchestration'
@requires: 'authenticated-user'
service OrchestrationService {
  action generateEmail(prompt : String, anonymize : Boolean) returns String;
}
