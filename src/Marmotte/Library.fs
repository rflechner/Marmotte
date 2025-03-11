namespace Marmotte

open System.Threading.Tasks
open Microsoft.AspNetCore.Components.Routing
open Microsoft.AspNetCore.Http
open Microsoft.AspNetCore.Routing.Patterns
open Microsoft.AspNetCore.Routing.Template
open Microsoft.AspNetCore.Routing

module RouteMatcher =
    
    let matches (ctx: HttpContext) (template:string) =
        ctx.Request.Path.Value = template

module Pipeline =

    type HttpRouteHandler = HttpContext -> Microsoft.AspNetCore.Http.IResult Task
    
    let inline endpoint method (template:string) (handler:HttpRouteHandler) =
        let pattern = RoutePatternFactory.Parse(template)
        let template = TemplateParser.Parse(template)
        let matcher = TemplateMatcher(template, RouteValueDictionary())

        fun (ctx: HttpContext) (next:System.Func<Task>) ->
            task {
                let routeValues = RouteValueDictionary()
                if ctx.Request.Method = method && matcher.TryMatch(ctx.Request.Path, routeValues) then
                    let! result = handler ctx
                    do! result.ExecuteAsync ctx
                else
                    do! next.Invoke()
            } :> Task

