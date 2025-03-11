namespace Marmotte

open System.Threading.Tasks
open Microsoft.AspNetCore.Http

module RouteMatcher =
    
    let matches (ctx: HttpContext) (template:string) =
        ctx.Request.Path.Value = template

module Pipeline =

    type HttpRouteHandler = HttpContext -> Microsoft.AspNetCore.Http.IResult Task
    
    let inline endpoint method (template:string) (handler:HttpRouteHandler) =
        fun (ctx: HttpContext) (next:System.Func<Task>) ->
            task {
                if ctx.Request.Method = method && RouteMatcher.matches ctx template then
                    let! result = handler ctx
                    do! result.ExecuteAsync ctx
                else
                    do! next.Invoke()
            } :> Task

