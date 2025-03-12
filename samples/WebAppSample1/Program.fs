namespace WebAppSample1
#nowarn "20"
open System
open System.Threading.Tasks
open Marmotte.Pipeline
open Microsoft.AspNetCore
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Http
open Microsoft.AspNetCore.Routing.Patterns
open Microsoft.AspNetCore.Routing.Template
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Microsoft.AspNetCore.OpenApi
open Scalar.AspNetCore

// https://learn.microsoft.com/fr-fr/aspnet/core/fundamentals/openapi/aspnetcore-openapi?view=aspnetcore-9.0&tabs=visual-studio
// https://learn.microsoft.com/en-us/aspnet/core/fundamentals/openapi/using-openapi-documents?view=aspnetcore-9.0#use-scalar-for-interactive-api-documentation

module Program =
    let exitCode = 0

    type UserDto = { Id: int; Name: string }
    
    let getUserById (id: int) : Task<UserDto option> =
        // Simule l'accès à une base de données ou autre stockage
        Task.FromResult(
            if id = 42 then Some { Id = 42; Name = "Douglas Adams" } else None
        )

    let getUserHandler (ctx: HttpContext) (next: RequestDelegate) =
        task {
            match ctx.Request.RouteValues.TryGetValue("id") with
            | true, (:? string as idStr) when System.Int32.TryParse(idStr) |> fst ->
                let id = int idStr
                let! user = getUserById id
                match user with
                | Some u ->
                    ctx.Response.StatusCode <- 200
                    ctx.Response.ContentType <- "application/json"
                    do! ctx.Response.WriteAsJsonAsync(u)
                | None ->
                    ctx.Response.StatusCode <- 404
                    do! ctx.Response.WriteAsync("User not found.")
            | _ ->
                ctx.Response.StatusCode <- 400
                do! ctx.Response.WriteAsync("Invalid ID provided.")
        }
    
    [<EntryPoint>]
    let main args =

        let builder = WebApplication.CreateBuilder(args)

        builder.Services.AddControllers()
        
        builder.Services.AddEndpointsApiExplorer()
        builder.Services.AddOpenApi()

        let app = builder.Build()

        if app.Environment.IsDevelopment()
        then
            app.MapOpenApi() |> ignore
            app.MapScalarApiReference() |> ignore
            //app.UseOpenApiExplorer() |> ignore
        
        app.UseHttpsRedirection()

        app.UseAuthorization()
        app.MapControllers()
        app.UseRouting()

        // app.Use(fun ctx next -> getUserHandler ctx next)

        app.MapGet("/users/{id:int}", Func<HttpContext, Task>(
            fun ctx ->
                task {
                    match ctx.Request.RouteValues.TryGetValue("id") with
                    | true, (:? string as idStr) when System.Int32.TryParse(idStr) |> fst ->
                        let id = int idStr
                        let! user = getUserById id
                        match user with
                        | Some u -> return Results.Json(u).ExecuteAsync(ctx)
                        | None -> return Results.NotFound("User not found").ExecuteAsync(ctx)
                    | _ ->
                        return Results.BadRequest("Invalid ID").ExecuteAsync(ctx)
                })) |> ignore
        
        let h =
            app.MapMethods("/toto/{id:int}", [HttpMethods.Get], Func<HttpContext, Task>(
                fun ctx ->
                    task {
                        match ctx.Request.RouteValues.TryGetValue("id") with
                        | true, (:? string as idStr) when System.Int32.TryParse(idStr) |> fst ->
                            let id = int idStr
                            let! user = getUserById id
                            match user with
                            | Some u -> return Results.Json(u).ExecuteAsync(ctx)
                            | None -> return Results.NotFound("Toto not found").ExecuteAsync(ctx)
                        | _ ->
                            return Results.BadRequest("Invalid ID").ExecuteAsync(ctx)
                    })).WithName("Toto").WithTags("toto").Produces<UserDto>(StatusCodes.Status200OK)
        
        
        let routeTemplate = "/pet/{id:int}"
        let pattern = RoutePatternFactory.Parse routeTemplate
        let template = TemplateParser.Parse(routeTemplate)
        let matcher = TemplateMatcher(template, Routing.RouteValueDictionary())
        let routeValues = Routing.RouteValueDictionary()
        matcher.TryMatch("/pet/1234", routeValues)

        
        let ep1 = endpoint HttpMethods.Get "/pet/{id:int}"
                    (fun ctx ->
                        task {
                            return Results.Ok("popo")
                        })
        app.Use ep1

        
        app.Run()

        exitCode